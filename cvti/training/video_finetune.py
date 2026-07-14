"""Unified temporal-model fine-tuner — VideoMAE or X3D (plan.md Phase 6).

Fine-tunes a pretrained video model on the CamNuvem robbery-vs-normal clips.
Both backends share `video_dataset.RobberyClipDataset`, so VideoMAE and X3D
train/eval on identical clips and their metrics are directly comparable.

    # smoke test (tiny, MPS/CPU ok)
    python -m cvti.training.video_finetune --backend videomae \
        --per-class-limit 3 --epochs 1 --batch 2

    # real run (cloud GPU)
    python -m cvti.training.video_finetune --backend videomae --epochs 8 --batch 4 --device cuda

X3D needs `pip install -r requirements-video.txt` (pytorchvideo); VideoMAE needs
transformers (already used by the inference path).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from cvti.training.video_dataset import DEFAULT_CLASS_MAP, DEFAULT_DATA_ROOT, RobberyClipDataset


def _auto_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Backends — both expose the same tiny interface to the training loop.
# ---------------------------------------------------------------------------

class _VideoMAEBackend:
    name = "videomae"
    default_model = "MCG-NJU/videomae-base-finetuned-kinetics"

    def __init__(self, base_model: str, num_labels: int, device: str, id2label: dict) -> None:
        import torch
        from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
        self.torch = torch
        self.device = device
        self.processor = VideoMAEImageProcessor.from_pretrained(base_model)
        self.model = VideoMAEForVideoClassification.from_pretrained(
            base_model, num_labels=num_labels, ignore_mismatched_sizes=True,
            id2label=id2label, label2id={v: k for k, v in id2label.items()},
        ).to(device)

    def prepare(self, clips):
        # Process each clip (list of RGB frames) then stack to [B,T,C,H,W].
        pv = [self.processor(list(clip), return_tensors="pt")["pixel_values"] for clip in clips]
        return {"pixel_values": self.torch.cat(pv, 0).to(self.device)}

    def logits(self, inputs):
        return self.model(**inputs).logits

    def parameters(self):
        return self.model.parameters()

    def freeze_backbone(self):
        # Train only the classifier head — the base is a strong pretrained
        # encoder, and 220 clips can't fine-tune 86M params without overfitting.
        for p in self.model.videomae.parameters():
            p.requires_grad = False
        for p in self.model.classifier.parameters():
            p.requires_grad = True

    def set_train(self, on: bool):
        self.model.train(on)

    def save(self, out: Path):
        self.model.save_pretrained(out)
        self.processor.save_pretrained(out)


class _X3DBackend:
    name = "x3d"
    default_model = "x3d_s"

    def __init__(self, base_model: str, num_labels: int, device: str, id2label: dict) -> None:
        import torch
        try:
            from pytorchvideo.models.hub import x3d_m, x3d_s, x3d_xs
        except ImportError as exc:  # noqa: BLE001
            raise SystemExit("X3D needs pytorchvideo: pip install -r requirements-video.txt") from exc
        self.torch = torch
        self.device = device
        loaders = {"x3d_xs": x3d_xs, "x3d_s": x3d_s, "x3d_m": x3d_m}
        model = loaders[base_model](pretrained=True)
        # Swap the Kinetics-400 projection head for a num_labels head.
        head = model.blocks[-1]
        in_features = head.proj.in_features
        head.proj = torch.nn.Linear(in_features, num_labels)
        self.model = model.to(device)

    def prepare(self, clips):
        from cvti.video_action_model import _frames_to_x3d_tensor
        # _frames_to_x3d_tensor -> [C,T,H,W]; stack to [B,C,T,H,W].
        tensors = [_frames_to_x3d_tensor(list(clip), self.torch) for clip in clips]
        return {"x": self.torch.stack(tensors, 0).to(self.device)}

    def logits(self, inputs):
        return self.model(inputs["x"])

    def parameters(self):
        return self.model.parameters()

    def freeze_backbone(self):
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.blocks[-1].proj.parameters():
            p.requires_grad = True

    def set_train(self, on: bool):
        self.model.train(on)

    def save(self, out: Path):
        self.torch.save(self.model.state_dict(), out / "x3d_finetuned.pt")


def _make_backend(name: str, base_model: str, num_labels: int, device: str, id2label: dict):
    if name == "videomae":
        return _VideoMAEBackend(base_model, num_labels, device, id2label)
    if name == "x3d":
        return _X3DBackend(base_model, num_labels, device, id2label)
    raise SystemExit(f"unknown backend: {name}")


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def _batches(n: int, batch: int, shuffle: bool):
    idx = list(range(n))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, n, batch):
        yield idx[i:i + batch]


def _load(ds: RobberyClipDataset, indices: list[int]):
    clips, labels = [], []
    for i in indices:
        frames, label = ds[i]
        clips.append(frames)
        labels.append(label)
    return clips, labels


def _evaluate(backend, torch, ds: RobberyClipDataset, batch: int, pos_label: int = 1) -> dict:
    backend.set_train(False)
    tp = fp = tn = fn = 0
    correct = total = 0
    with torch.no_grad():
        for chunk in _batches(len(ds), batch, shuffle=False):
            clips, labels = _load(ds, chunk)
            preds = backend.logits(backend.prepare(clips)).argmax(dim=-1).tolist()
            for p, y in zip(preds, labels):
                total += 1
                correct += int(p == y)
                if y == pos_label:
                    tp += int(p == pos_label); fn += int(p != pos_label)
                else:
                    fp += int(p == pos_label); tn += int(p != pos_label)
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"accuracy": round(correct / total, 4) if total else 0.0,
            "recall_theft": round(recall, 4), "fpr_normal": round(fpr, 4),
            "n": total, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune VideoMAE/X3D on CamNuvem (Phase 6).")
    p.add_argument("--backend", choices=("videomae", "x3d"), default="videomae")
    p.add_argument("--base-model", default="", help="Blank = backend default.")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--per-class-limit", type=int, default=0, help="Cap clips/class (0=all). Use small for smoke tests.")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="Train only the classifier head (recommended for a small "
                        "dataset — the pretrained encoder overfits fast otherwise).")
    p.add_argument("--patience", type=int, default=0,
                   help="Early-stop if val recall_theft hasn't improved in N epochs (0=off).")
    p.add_argument("--device", default="")
    p.add_argument("--out", default="runs/video_finetune")
    args = p.parse_args()

    import torch

    device = args.device or _auto_device()
    class_map = DEFAULT_CLASS_MAP
    id2label = {v: k for k, v in class_map.items()}
    limit = args.per_class_limit or None

    train_ds = RobberyClipDataset(args.data_root, "training", frames=args.frames,
                                  class_map=class_map, per_class_limit=limit)
    test_ds = RobberyClipDataset(args.data_root, "test", frames=args.frames,
                                 class_map=class_map, per_class_limit=limit)
    if len(train_ds) == 0:
        raise SystemExit(f"No training clips under {args.data_root}. Check the dataset path.")
    print(f"[finetune] backend={args.backend} device={device} "
          f"train={len(train_ds)} test={len(test_ds)} frames={args.frames}")

    base_model = args.base_model or _backend_default(args.backend)
    backend = _make_backend(args.backend, base_model, len(class_map), device, id2label)
    if args.freeze_backbone:
        backend.freeze_backbone()
        print("[finetune] backbone frozen — training classifier head only")
    trainable = [p for p in backend.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    out_dir = Path(args.out) / args.backend
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_recall = -1.0
    epochs_since_best = 0

    for epoch in range(1, args.epochs + 1):
        backend.set_train(True)
        t0 = time.time()
        running = 0.0
        steps = 0
        for chunk in _batches(len(train_ds), args.batch, shuffle=True):
            clips, labels = _load(train_ds, chunk)
            inputs = backend.prepare(clips)
            logits = backend.logits(inputs)
            loss = loss_fn(logits, torch.tensor(labels, device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
            steps += 1
        metrics = _evaluate(backend, torch, test_ds, args.batch) if len(test_ds) else {}
        row = {"epoch": epoch, "train_loss": round(running / max(1, steps), 4),
               "secs": round(time.time() - t0, 1), **metrics}
        history.append(row)
        print(f"[epoch {epoch}] loss={row['train_loss']} "
              + " ".join(f"{k}={metrics[k]}" for k in ("accuracy", "recall_theft", "fpr_normal") if k in metrics))
        if metrics.get("recall_theft", -1) > best_recall:
            best_recall = metrics.get("recall_theft", -1)
            epochs_since_best = 0
            backend.save(out_dir)
            print(f"[epoch {epoch}] saved best (recall_theft={best_recall})")
        else:
            epochs_since_best += 1
        # Persist metrics EVERY epoch so a long/unattended run stays observable
        # and survives an interruption.
        (out_dir / "metrics.json").write_text(json.dumps(
            {"backend": args.backend, "base_model": base_model, "device": device,
             "class_map": class_map, "epochs_planned": args.epochs,
             "best_recall_theft": best_recall, "history": history}, indent=2))
        if args.patience and epochs_since_best >= args.patience:
            print(f"[finetune] early stop: no val improvement in {args.patience} epochs")
            break

    print(f"[finetune] done. checkpoint + metrics in {out_dir}")


def _backend_default(name: str) -> str:
    return {"videomae": _VideoMAEBackend.default_model, "x3d": _X3DBackend.default_model}[name]


if __name__ == "__main__":
    main()
