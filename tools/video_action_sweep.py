"""video_action_sweep.py — what does the video-action theft threshold actually buy?

    python tools/video_action_sweep.py            # score every clip, then sweep
    python tools/video_action_sweep.py --report   # re-sweep already-scored clips

The 24 Sep KPI run measured a 25.9% false-positive rate, and ALL 44 of those
false positives came from one rule: `video_theft_candidate`. The same model
finds only 46.9% of real theft. `DEFAULT_RAW_CONFIDENCE_THRESHOLD` is 0.05 —
a 5%% "theft" score is enough to raise a candidate, and every candidate buys
a ~12s VLM verification on the pilot's 4-core box.

This scores each clip ONCE (the model's peak theft probability over the same
windows the engine analyses) and then sweeps the threshold offline, so the
recall-vs-false-alarm trade is a measurement rather than an opinion. No
retraining, no new footage, no VLM.

Resumable: per-clip scores land in runs/eval/video_action/scores.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "runs" / "eval" / "video_action"
SCORES = OUT / "scores.jsonl"

# The engine's own window geometry (PerCameraState: va_fps, va_window_seconds,
# va_frames, va_cooldown) so a sweep result means something for the product.
VA_FPS = 5.0
WINDOW_S = 4.0
COOLDOWN_S = 2.0
MAX_SECONDS = 30.0          # the eval harness's per-clip budget


def clip_windows(path: Path, max_seconds: float = MAX_SECONDS) -> list:
    """Frames grouped exactly as the engine's rolling analysis would see them."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / VA_FPS)))
    frames, i = [], 0
    want = round(max_seconds * VA_FPS)
    while len(frames) < want:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        if i % step:
            continue
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    per_window = max(2, int(WINDOW_S * VA_FPS))
    stride = max(1, int(COOLDOWN_S * VA_FPS))
    return [frames[s:s + per_window] for s in range(0, max(1, len(frames) - per_window + 1), stride)]


def theft_confidence(preds) -> float:
    """Peak confidence over labels the hybrid bridge calls theft."""
    from cvti.video_action_hybrid import classify_action_label
    best = 0.0
    for p in preds:
        if classify_action_label(p.label) == "theft_candidate":
            best = max(best, float(p.confidence))
    return best


def score_clips(clips: list[tuple[str, bool]]) -> None:
    from cvti.video_action_model import VideoMAEActionModel
    done = set()
    if SCORES.exists():
        done = {json.loads(l)["path"] for l in SCORES.read_text().splitlines() if l.strip()}
    todo = [(p, t) for p, t in clips if p not in done]
    print(f"{len(done)} already scored, {len(todo)} to go")
    if not todo:
        return
    model = VideoMAEActionModel(str(ROOT / "runs" / "video_finetune" / "videomae"))
    model.load()
    OUT.mkdir(parents=True, exist_ok=True)
    with SCORES.open("a") as fh:
        for n, (rel, is_theft) in enumerate(todo, 1):
            t0 = time.monotonic()
            try:
                windows = clip_windows(ROOT / rel)
                confs = [theft_confidence(model.predict_frames(w)) for w in windows if len(w) >= 2]
            except Exception as exc:  # noqa: BLE001 - one bad clip must not end the sweep
                fh.write(json.dumps({"path": rel, "is_theft": is_theft,
                                     "error": str(exc)[:200]}) + "\n"); fh.flush()
                print(f"  [{n}/{len(todo)}] ERROR {rel}: {str(exc)[:80]}")
                continue
            peak = max(confs) if confs else 0.0
            fh.write(json.dumps({"path": rel, "is_theft": is_theft, "peak": peak,
                                 "windows": len(confs),
                                 "confs": [round(c, 4) for c in confs]}) + "\n")
            fh.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"  [{n}/{len(todo)}] {rel.split('/')[-1][:40]} peak={peak:.3f} "
                      f"({time.monotonic() - t0:.1f}s)")


def sweep() -> None:
    rows = [json.loads(l) for l in SCORES.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "peak" in r]
    normals = [r for r in rows if not r["is_theft"]]
    thefts = [r for r in rows if r["is_theft"]]
    print(f"\nscored: {len(normals)} normals, {len(thefts)} theft clips\n")
    print(f"{'threshold':>9} {'false positives':>16} {'theft recall':>14} "
          f"{'VLM calls / 170 normals':>24}")
    best = None
    for t in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        fp = sum(1 for r in normals if r["peak"] >= t)
        tp = sum(1 for r in thefts if r["peak"] >= t)
        fpr = fp / len(normals) if normals else 0.0
        rec = tp / len(thefts) if thefts else 0.0
        mark = ""
        if fpr <= 0.05 and (best is None or rec > best[2]):
            best = (t, fpr, rec); mark = "  <- meets the <=5% FP target"
        print(f"{t:>9.2f} {fp:>6}/{len(normals):<4} = {fpr*100:>5.1f}% "
              f"{tp:>5}/{len(thefts):<4} = {rec*100:>5.1f}% {fp:>18}{mark}")
    print("\ncurrent shipped threshold: 0.05 "
          "(cvti/video_action_hybrid.py DEFAULT_RAW_CONFIDENCE_THRESHOLD)")
    if best:
        print(f"best threshold meeting the <=5% false-positive target: {best[0]:.2f} "
              f"-> {best[2]*100:.1f}% theft recall")
    else:
        print("NO threshold reaches the <=5% false-positive target: this model "
              "cannot separate theft from normal CCTV on this footage.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", action="store_true", help="skip scoring; sweep what exists")
    args = ap.parse_args()
    if not args.report:
        man = json.loads((ROOT / "configs/eval/kpi_manifest_v1.json").read_text())["rows"]
        clips = [(c["path"], False) for c in man["false_positive"]["clips"]]
        clips += [(c["path"], True) for c in man["theft"]["clips"]]
        score_clips(clips)
    sweep()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
