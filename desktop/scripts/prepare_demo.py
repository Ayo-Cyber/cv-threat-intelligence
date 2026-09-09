"""Prepare local preview media; recordings are never committed with this UI."""
import argparse
from pathlib import Path
import shutil
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument('--assets', type=Path)
args = parser.parse_args()
target = Path(__file__).resolve().parents[1] / 'public' / 'demo'
target.mkdir(parents=True, exist_ok=True)
source = args.assets or args.repo / 'data' / 'test_clips'
for name in ['empty_warehouse', 'theft_yt_01', 'theft_shop_01', 'normal_street_01']:
    path = source / (name + '.mp4')
    if not path.exists():
        raise SystemExit(f'Missing demo recording: {path}')
    shutil.copy2(path, target / path.name)
    video = cv2.VideoCapture(str(path))
    video.set(cv2.CAP_PROP_POS_FRAMES, 20)
    ok, frame = video.read()
    video.release()
    if not ok or not cv2.imwrite(str(target / (name + '-frame.jpg')), frame):
        raise SystemExit(f'Could not make preview frame for {path}')
    print(f'Prepared {name}')
