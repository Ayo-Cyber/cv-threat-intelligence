"""tools/draw_zones.py — Interactive GUI tool to draw custom polygon zones on video frames.

Usage:
    python tools/draw_zones.py --source data/test_clips/theft_shop_01.mp4 --out configs/my_custom_zones.json

Controls:
    - Left Click  : Add a point (vertex) to the current polygon.
    - Right Click : Finish the current polygon and save zone.
    - Key 'c'     : Clear current points.
    - Key 's'     : Save all zones to JSON file and exit.
    - Key 'q'     : Quit without saving.
"""

import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np


points: list[tuple[int, int]] = []
completed_zones: list[dict] = []
current_zone_name = "zone_1"


def mouse_callback(event, x, y, flags, param):
    global points, completed_zones, current_zone_name

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point added: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(points) >= 3:
            name = input(f"\nEnter name for this zone [default: {current_zone_name}]: ").strip()
            if not name:
                name = current_zone_name

            completed_zones.append({
                "name": name,
                "kind": "shelf",
                "dwell_alert_seconds": 5.0,
                "polygon": [list(p) for p in points],
                "anchors": ["ALL"]
            })
            print(f"✅ Zone '{name}' saved with {len(points)} points!\n")

            # Prepare for next zone
            points = []
            current_zone_name = f"zone_{len(completed_zones) + 1}"
            print("Click on the image to start drawing the NEXT zone (or press 's' to save & exit).")
        else:
            print("⚠️ A polygon must have at least 3 points before right-clicking to finish.")


def main():
    global points, completed_zones, current_zone_name

    parser = argparse.ArgumentParser(description="Interactive Polygon Zone Drawer")
    parser.add_argument("--source", required=True, help="Path to video file, RTSP URL, or webcam index (e.g. 0)")
    parser.add_argument("--out", default="configs/my_custom_zones.json", help="Path to save the output JSON zone file")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"❌ Error: Could not open source '{args.source}'")
        sys.exit(1)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"❌ Error: Could not read frame from source '{args.source}'")
        sys.exit(1)

    window_name = "CVTI Zone Drawer - Left Click: Add Point | Right Click: Finish Zone | 's': Save | 'c': Clear | 'q': Quit"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n" + "=" * 70)
    print(" 🎨 INTERACTIVE ZONE DRAWER TOOL ")
    print("=" * 70)
    print(" 1. Left Click on the video frame to place polygon points.")
    print(" 2. Right Click when done drawing a zone to name and finish it.")
    print(" 3. Repeat for as many zones as you want.")
    print(" 4. Press 's' to save the zone JSON file and exit.")
    print(" 5. Press 'c' to clear current un-saved points.")
    print(" 6. Press 'q' to quit without saving.")
    print("=" * 70 + "\n")

    while True:
        canvas = frame.copy()

        # Draw already completed zones in Green
        for zone in completed_zones:
            pts = np.array(zone["polygon"], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            # Label near first point
            first_pt = zone["polygon"][0]
            cv2.putText(canvas, zone["name"], (first_pt[0], first_pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw current active points in Red
        if len(points) > 0:
            for p in points:
                cv2.circle(canvas, p, 5, (0, 0, 255), -1)

            if len(points) > 1:
                pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts], isClosed=False, color=(0, 0, 255), thickness=2)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("s"):
            if not completed_zones and len(points) >= 3:
                # Auto-finish if points exist
                completed_zones.append({
                    "name": current_zone_name,
                    "kind": "shelf",
                    "dwell_alert_seconds": 5.0,
                    "polygon": [list(p) for p in points],
                    "anchors": ["ALL"]
                })

            if not completed_zones:
                print("⚠️ No zones were completed. Nothing to save.")
                break

            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({"zones": completed_zones}, indent=2), encoding="utf-8")

            print("\n" + "🎉 SUCCESS!" + "=" * 60)
            print(f"Saved {len(completed_zones)} zone(s) to: {out_path.resolve()}")
            print("\nNow run your test with your custom drawn zones:")
            print(f"python retail_zones.py --source \"{args.source}\" --zones \"{args.out}\" --rules configs/banking_zones_v1.json --simulate-time 21:00 --show")
            print("=" * 70 + "\n")
            break

        elif key == ord("c"):
            points = []
            print("Cleared current points.")

        elif key == ord("q"):
            print("Exited without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
