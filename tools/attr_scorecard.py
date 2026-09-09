"""attr_scorecard.py — score the W3 open-vocab router on the attribute manifest.

    python tools/attr_scorecard.py                 # all attributes
    python tools/attr_scorecard.py --attrs cap,backpack

The charter's W3 acceptance: attribute rules answered by a REAL detector with
real boxes, no false positive on a bare head, 0 hallucinated attributes, and
<100ms per answer. This runs OpenVocabDetector over every manifest clip
(FRAMES_PER_CLIP frames spread across it; an attribute counts as PRESENT when
any frame scores >= the detector floor) and prints per-attribute precision /
recall with Wilson intervals, the hallucination count on absent-labeled clips,
and measured ms per frame. Results land in runs/eval/attr/scorecard.{json,md}.

Labels come from configs/eval/attr_manifest_v1.json — human-verified from
contact sheets, never from a model. A clip that omits an attribute key is
excluded from that attribute's scoring (uncertain label ≠ a negative).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "eval" / "attr_manifest_v1.json"
OUT = ROOT / "runs" / "eval" / "attr"

FRAMES_PER_CLIP = 5


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def clip_frames(path: Path, count: int):
    import cv2
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    picks = sorted({int(total * (i + 0.5) / count) for i in range(count)}) \
        if total > count else None
    out, idx = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if picks is None or idx in picks:
            out.append(fr)
        idx += 1
        if picks is not None and len(out) == count:
            break
    cap.release()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--attrs", default="", help="comma-separated (default: all)")
    ap.add_argument("--manifest", default=str(MANIFEST))
    args = ap.parse_args()

    doc = json.loads(Path(args.manifest).read_text())
    phrases: dict = doc["phrases"]
    wanted = [a.strip() for a in args.attrs.split(",") if a.strip()] or \
             [a for a in phrases
              if any(a in c for c in doc["clips"])]      # only labeled attrs

    from cvti.detector.openvocab import OpenVocabDetector, floor_for
    det = OpenVocabDetector()

    per_clip, ms = [], []
    for c in doc["clips"]:
        path = ROOT / c["path"]
        if not path.exists():
            continue
        frames = clip_frames(path, FRAMES_PER_CLIP)
        best: dict[str, float] = {a: 0.0 for a in wanted}
        for fr in frames:
            dets = det.detect(fr, [phrases[a] for a in wanted])
            if dets is None:
                raise SystemExit("detector unavailable — cannot score "
                                 f"({det.load_error})")
            ms.append(det.last_ms)
            for d in dets:
                for a in wanted:
                    if d["phrase"] == phrases[a]:
                        best[a] = max(best[a], d["score"])
        per_clip.append({"path": c["path"],
                         "labels": {a: c.get(a) for a in wanted},
                         "scores": {a: round(best[a], 3) for a in wanted}})
        print(f"  {path.name:<28} " +
              " ".join(f"{a}:{best[a]:.2f}" for a in wanted), flush=True)

    rows, halluc_total = [], 0
    for a in wanted:
        tp = fp = fn = tn = 0
        for r in per_clip:
            label = r["labels"][a]
            if label is None:
                continue
            hit = r["scores"][a] >= floor_for(phrases[a])
            if label and hit:
                tp += 1
            elif label and not hit:
                fn += 1
            elif not label and hit:
                fp += 1
            else:
                tn += 1
        halluc_total += fp
        prec_lo, _ = wilson(tp, tp + fp)
        rec_lo, _ = wilson(tp, tp + fn)
        rows.append({"attr": a, "phrase": phrases[a], "tp": tp, "fp": fp,
                     "fn": fn, "tn": tn,
                     "precision": tp / (tp + fp) if tp + fp else None,
                     "recall": tp / (tp + fn) if tp + fn else None,
                     "precision_lo": round(prec_lo, 3),
                     "recall_lo": round(rec_lo, 3)})

    ms.sort()
    med = ms[len(ms) // 2] if ms else 0.0
    print(f"\n{'attr':<10} {'phrase':<26} {'P':>7} {'R':>7} "
          f"{'tp':>3} {'fp':>3} {'fn':>3} {'tn':>3}")
    for r in rows:
        pr = f"{100 * r['precision']:.0f}%" if r["precision"] is not None else "  —"
        rc = f"{100 * r['recall']:.0f}%" if r["recall"] is not None else "  —"
        print(f"{r['attr']:<10} {r['phrase']:<26} {pr:>7} {rc:>7} "
              f"{r['tp']:>3} {r['fp']:>3} {r['fn']:>3} {r['tn']:>3}")
    print(f"\nhallucinated attributes (fp on absent-labeled clips): {halluc_total}")
    print(f"latency: median {med:.0f}ms per frame over {len(ms)} calls "
          f"(SLO <100ms) — floors: object {det.min_score}, worn 0.45")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scorecard.json").write_text(json.dumps(
        {"generated_at": time.time(), "manifest": str(args.manifest),
         "frames_per_clip": FRAMES_PER_CLIP, "floor": det.min_score,
         "median_ms": round(med, 1), "rows": rows, "clips": per_clip}, indent=1))
    print(f"\nwritten: {OUT.relative_to(ROOT)}/scorecard.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
