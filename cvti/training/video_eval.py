"""Evaluate a trained video model — full metrics, not just training-loop numbers.

Loads a saved checkpoint and reports precision / recall / FPR / F1 / balanced
accuracy + a confusion matrix on a held-out set. Supports a STRATIFIED split
(pooled + class-balanced) so the test set isn't class-skewed relative to train —
the CamNuvem native split is inverted, which makes raw metrics misleading.

    # honest metrics on a stratified test split
    python -m cvti.training.video_eval --checkpoint runs/video_finetune/videomae \
        --stratified --val-fraction 0.25 --seed 1234

    # metrics on CamNuvem's own test split
    python -m cvti.training.video_eval --checkpoint runs/video_finetune/videomae
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvti.training.video_dataset import (
    DEFAULT_CLASS_MAP, DEFAULT_DATA_ROOT, RobberyClipDataset, pool_clips, stratified_split,
)
from cvti.logging_setup import get_logger

log = get_logger(__name__)


def _auto_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _metrics(preds: list[int], labels: list[int], pos: int = 1) -> dict:
    tp = sum(1 for p, y in zip(preds, labels) if y == pos and p == pos)
    fn = sum(1 for p, y in zip(preds, labels) if y == pos and p != pos)
    fp = sum(1 for p, y in zip(preds, labels) if y != pos and p == pos)
    tn = sum(1 for p, y in zip(preds, labels) if y != pos and p != pos)
    n = len(labels)
    recall = tp / (tp + fn) if (tp + fn) else 0.0          # theft caught
    precision = tp / (tp + fp) if (tp + fp) else 0.0        # of theft alerts, how many real
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0             # normals wrongly flagged
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": n, "accuracy": round((tp + tn) / n, 4) if n else 0.0,
        "balanced_acc": round((recall + specificity) / 2, 4),
        "recall_theft": round(recall, 4), "precision_theft": round(precision, 4),
        "f1_theft": round(f1, 4), "fpr_normal": round(fpr, 4),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def main() -> None:
    # Entrypoint: without this, log records have no handler and
    # anything below WARNING is silently discarded.
    from cvti.logging_setup import setup_logging
    setup_logging(component="argus-video-eval")
    p = argparse.ArgumentParser(description="Evaluate a trained video model (Phase 5 metrics).")
    p.add_argument("--checkpoint", required=True, help="Saved checkpoint dir (from video_finetune).")
    p.add_argument("--backend", choices=("videomae", "x3d"), default="videomae")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--stratified", action="store_true", help="Pool + class-balanced split (recommended).")
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="")
    p.add_argument("--out", default="")
    args = p.parse_args()

    import torch
    from cvti.training.video_finetune import _backend_default, _batches, _load, _make_backend

    device = args.device or _auto_device()
    class_map = DEFAULT_CLASS_MAP
    id2label = {v: k for k, v in class_map.items()}

    if args.stratified:
        _, test_items = stratified_split(pool_clips(args.data_root, class_map=class_map),
                                         val_fraction=args.val_fraction, seed=args.seed)
        test_ds = RobberyClipDataset(items=test_items, frames=args.frames, class_map=class_map)
        split_desc = f"stratified(val_fraction={args.val_fraction}, seed={args.seed})"
    else:
        test_ds = RobberyClipDataset(args.data_root, "test", frames=args.frames, class_map=class_map)
        split_desc = "camnuvem-native-test"

    from collections import Counter
    log.info(f"[eval] checkpoint={args.checkpoint} device={device} test={len(test_ds)} "
          f"split={split_desc} class_counts={dict(Counter(test_ds.labels()))}")

    # Load the saved checkpoint (from_pretrained reads the fine-tuned weights).
    backend = _make_backend(args.backend, args.checkpoint, len(class_map), device, id2label)
    backend.set_train(False)

    preds: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for chunk in _batches(len(test_ds), args.batch, shuffle=False):
            clips, ys = _load(test_ds, chunk)
            batch_preds = backend.logits(backend.prepare(clips)).argmax(dim=-1).tolist()
            preds.extend(batch_preds)
            labels.extend(ys)

    m = _metrics(preds, labels)
    log.info("\n=== metrics ===")
    for k in ("n", "accuracy", "balanced_acc", "recall_theft", "precision_theft", "f1_theft", "fpr_normal"):
        log.info(f"  {k:>16}: {m[k]}")
    c = m["confusion"]
    log.info("\n  confusion (rows=true, cols=pred):")
    log.info(f"                 pred_normal  pred_theft")
    log.info(f"    true_normal   {c['tn']:>10}  {c['fp']:>10}")
    log.info(f"    true_theft    {c['fn']:>10}  {c['tp']:>10}")

    report = {"checkpoint": args.checkpoint, "backend": args.backend, "split": split_desc,
              "device": device, **m}
    out = args.out or str(Path(args.checkpoint) / "eval.json")
    Path(out).write_text(json.dumps(report, indent=2))
    log.info(f"\n[eval] wrote {out}")
    if m["n"] < 60:
        log.info(f"[eval] NOTE: only {m['n']} test clips — encouraging but not a validated FPR; "
              "needs more normal/hard-negative footage before trusting it.")


if __name__ == "__main__":
    main()
