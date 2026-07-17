#!/usr/bin/env python3
"""Render a 2x2 multi-camera 'video wall' for demos.

Runs ONE shared YOLO detector + ONE shared fine-tuned VideoMAE across up to 4
clips, draws person boxes + a per-camera alert banner (red THEFT when the video
model flags it, green CLEAR otherwise), tiles them into a grid, and writes an
mp4. This is a visual demo artifact — the real pipeline is cvti.serving (headless).

    python tools/demo_multicam_grid.py --out runs/serving/multicam_grid.mp4 \
        --sources clipA.mp4 clipB.mp4 clipC.mp4 clipD.mp4 --max-frames 150
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np

RED = (40, 40, 220)
GREEN = (60, 170, 60)
WHITE = (255, 255, 255)


def main() -> None:
    p = argparse.ArgumentParser(description="2x2 multi-camera demo grid renderer.")
    p.add_argument("--sources", nargs="+", required=True, help="Up to 4 clip paths.")
    p.add_argument("--out", default="runs/serving/multicam_grid.mp4")
    p.add_argument("--weights", default="models/yolov8n.pt")
    p.add_argument("--video-model", default="runs/video_finetune/videomae")
    p.add_argument("--max-frames", type=int, default=150, help="Frames per camera (grid length).")
    p.add_argument("--tile", type=int, default=360, help="Tile size (px, square).")
    p.add_argument("--va-every", type=int, default=25, help="Run the video model every N frames.")
    p.add_argument("--va-frames", type=int, default=16)
    p.add_argument("--fps", type=float, default=25.0)
    args = p.parse_args()

    from ultralytics import YOLO
    from cvti.video_action_model import VideoMAEActionModel, sample_evenly

    sources = args.sources[:4]
    yolo = YOLO(args.weights)
    video = VideoMAEActionModel(args.video_model)
    print(f"[grid] {len(sources)} cameras | detector={args.weights} | video={args.video_model}")

    caps = [cv2.VideoCapture(s) for s in sources]
    buffers = [deque(maxlen=max(args.va_frames, 48)) for _ in sources]
    banners = [("CLEAR", 0.0) for _ in sources]     # (label, confidence)
    names = [Path(s).stem[:22] for s in sources]

    t = args.tile
    grid_w, grid_h = t * 2, t * 2
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (grid_w, grid_h))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    snap_saved = False
    for fi in range(args.max_frames):
        tiles = []
        for ci, cap in enumerate(caps):
            ok, frame = cap.read()
            if not ok:
                frame = np.zeros((t, t, 3), dtype=np.uint8)
            else:
                buffers[ci].append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # person boxes
                res = yolo.predict(frame, classes=[0], conf=0.4, verbose=False)[0]
                if res.boxes is not None:
                    for b in res.boxes.xyxy.tolist():
                        x1, y1, x2, y2 = (int(v) for v in b)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                # video model every va_every frames -> sticky banner
                if fi % args.va_every == 0 and len(buffers[ci]) >= args.va_frames:
                    clip = sample_evenly(list(buffers[ci]), count=args.va_frames)
                    preds = video.predict_frames(clip, top_k=2)
                    top = preds[0]
                    if top.label.lower() == "theft":
                        banners[ci] = ("THEFT", top.confidence)
                    else:
                        banners[ci] = ("CLEAR", top.confidence)

            tile = cv2.resize(frame, (t, t))
            label, conf = banners[ci]
            colour = RED if label == "THEFT" else GREEN
            cv2.rectangle(tile, (0, 0), (t, 30), colour, -1)
            text = f"{names[ci]}   {'ALERT: THEFT' if label == 'THEFT' else 'clear'} {conf:.2f}"
            cv2.putText(tile, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
            cv2.rectangle(tile, (0, 0), (t - 1, t - 1), (60, 60, 60), 1)
            tiles.append(tile)

        while len(tiles) < 4:
            tiles.append(np.zeros((t, t, 3), dtype=np.uint8))
        top_row = np.hstack([tiles[0], tiles[1]])
        bot_row = np.hstack([tiles[2], tiles[3]])
        grid = np.vstack([top_row, bot_row])
        writer.write(grid)
        if not snap_saved and fi > 0 and any(b[0] == "THEFT" for b in banners):
            cv2.imwrite(str(Path(args.out).with_suffix(".png")), grid)
            snap_saved = True

    for cap in caps:
        cap.release()
    writer.release()
    if not snap_saved:
        # save the last grid frame if no alert ever fired
        cv2.imwrite(str(Path(args.out).with_suffix(".png")), grid)
    print(f"[grid] wrote {args.out} ({args.max_frames} frames) + snapshot .png")


if __name__ == "__main__":
    main()
