#!/usr/bin/env python3
"""Live multi-camera dashboard (4-6 feeds) — detection + video model at once.

Tiles up to 6 sources into one grid, runs ONE shared YOLO detector BATCHED
across all cameras each tick (that's the "inference at once"), draws person
boxes + a per-camera red/green THEFT|CLEAR banner from the fine-tuned VideoMAE,
and shows it live (--show) and/or writes an mp4. Sources can be clips, webcam
indices, or rtsp:// URLs. Clips loop so the dashboard runs continuously.

    # live window on your machine (press q to quit)
    python tools/demo_dashboard.py --show --sources cam1.mp4 cam2.mp4 cam3.mp4 cam4.mp4

    # headless: render a grid mp4 instead
    python tools/demo_dashboard.py --out runs/serving/dashboard.mp4 --seconds 12 \
        --sources cam1.mp4 cam2.mp4 cam3.mp4 cam4.mp4 cam5.mp4 cam6.mp4
"""
from __future__ import annotations

import argparse
import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

RED = (40, 40, 220)
GREEN = (60, 170, 60)
WHITE = (255, 255, 255)
HEADER_H = 34


def _open(source: str):
    src = int(source) if str(source).isdigit() else source
    return cv2.VideoCapture(src)


def main() -> None:
    p = argparse.ArgumentParser(description="Live 4-6 camera detection dashboard.")
    p.add_argument("--site-config", default="", help="Read sources from a serving site-config JSON.")
    p.add_argument("--sources", nargs="+", help="Or list 2-6 clips / webcam idx / rtsp URLs directly.")
    p.add_argument("--show", action="store_true", help="Live window (needs a display; press q to quit).")
    p.add_argument("--out", default="", help="Also/instead write a grid mp4 here.")
    p.add_argument("--weights", default="models/yolov8n.pt")
    p.add_argument("--video-model", default="runs/video_finetune/videomae")
    p.add_argument("--no-video", action="store_true", help="Detection only (skip the theft banner model).")
    p.add_argument("--tile", type=int, default=360)
    p.add_argument("--va-every", type=int, default=25, help="Score the video model every N ticks.")
    p.add_argument("--va-frames", type=int, default=16)
    p.add_argument("--seconds", type=float, default=12.0, help="Headless run length (ignored with --show).")
    p.add_argument("--fps", type=float, default=20.0)
    args = p.parse_args()

    from ultralytics import YOLO
    site_names: list[str] | None = None
    if args.site_config:
        import json
        cams = json.loads(Path(args.site_config).read_text())["cameras"][:6]
        sources = [str(c["source"]) for c in cams]
        site_names = [str(c.get("id", Path(str(c["source"])).stem)) for c in cams]
    elif args.sources:
        sources = args.sources[:6]
    else:
        raise SystemExit("give --site-config or --sources")
    n = len(sources)
    yolo = YOLO(args.weights)
    video = None
    if not args.no_video:
        from cvti.video_action_model import VideoMAEActionModel
        video = VideoMAEActionModel(args.video_model)
    print(f"[dash] {n} cameras | detector={args.weights} | video={'off' if args.no_video else args.video_model}")

    caps = [_open(s) for s in sources]
    buffers = [deque(maxlen=max(args.va_frames, 48)) for _ in sources]
    banners = [("clear", 0.0) for _ in sources]
    names = site_names[:] if site_names else [
        Path(str(s)).stem[:20] if not str(s).isdigit() else f"webcam{s}" for s in sources]
    names = [nm[:20] for nm in names]

    cols = 2 if n <= 4 else 3
    rows = math.ceil(n / cols)
    t = args.tile
    grid_w, grid_h = t * cols, t * rows + HEADER_H

    writer = None
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (grid_w, grid_h))

    from cvti.video_action_model import sample_evenly  # noqa: E402
    total_alerts = 0
    tick = 0
    t_start = time.time()
    max_ticks = None if args.show else int(args.seconds * args.fps)

    while True:
        frames = []
        for ci, cap in enumerate(caps):
            ok, frame = cap.read()
            if not ok:                       # loop clips so the dashboard runs continuously
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if not ok:
                frame = np.zeros((t, t, 3), dtype=np.uint8)
            frames.append(frame)
            buffers[ci].append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # ONE batched detector pass across all cameras.
        results = yolo.predict(frames, classes=[0], conf=0.4, verbose=False)

        tiles = []
        for ci, (frame, res) in enumerate(zip(frames, results)):
            draw = frame.copy()
            if res.boxes is not None:
                for b in res.boxes.xyxy.tolist():
                    x1, y1, x2, y2 = (int(v) for v in b)
                    cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 200, 255), 2)
            if video is not None and tick % args.va_every == 0 and len(buffers[ci]) >= args.va_frames:
                preds = video.predict_frames(sample_evenly(list(buffers[ci]), count=args.va_frames), top_k=2)
                top = preds[0]
                was = banners[ci][0]
                banners[ci] = (("THEFT", top.confidence) if top.label.lower() == "theft"
                               else ("clear", top.confidence))
                if banners[ci][0] == "THEFT" and was != "THEFT":
                    total_alerts += 1
            tile = cv2.resize(draw, (t, t))
            label, conf = banners[ci]
            colour = RED if label == "THEFT" else GREEN
            cv2.rectangle(tile, (0, 0), (t, 26), colour, -1)
            cv2.putText(tile, f"{names[ci]}  {'ALERT: THEFT' if label=='THEFT' else 'clear'} {conf:.2f}",
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, WHITE, 1, cv2.LINE_AA)
            cv2.rectangle(tile, (0, 0), (t - 1, t - 1), (60, 60, 60), 1)
            tiles.append(tile)
        while len(tiles) < cols * rows:
            tiles.append(np.zeros((t, t, 3), dtype=np.uint8))

        grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
        header = np.full((HEADER_H, grid_w, 3), 30, dtype=np.uint8)
        fps = (tick + 1) / max(time.time() - t_start, 1e-6)
        cv2.putText(header, f"CVTI  |  {n} cameras  |  alerts: {total_alerts}  |  {fps:.1f} fps",
                    (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
        dash = np.vstack([header, grid])

        if writer is not None:
            writer.write(dash)
        if args.show:
            cv2.imshow("CVTI dashboard", dash)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        tick += 1
        if max_ticks is not None and tick >= max_ticks:
            break

    for cap in caps:
        cap.release()
    if writer is not None:
        cv2.imwrite(str(Path(args.out).with_suffix(".png")), dash)
        writer.release()
        print(f"[dash] wrote {args.out} + snapshot .png")
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
