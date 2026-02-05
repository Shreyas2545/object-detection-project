import os
import cv2
from pathlib import Path
import numpy as np

"""
Extract frames from videos in data/ai_videos and write them into a frame dataset
suitable for training the image-level detector under data/ai_detector/{train,val}/{ai,real}/

Usage:
  python src/prepare_frame_dataset.py --src_dir data/ai_videos --out_dir data/ai_detector --frames_per_video 8
"""


def extract_frames_from_video(video_path, out_dir, frames_per_video=8, size=(224,224)):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return 0
    indices = np.linspace(0, total-1, frames_per_video, dtype=int)
    saved = 0
    name_base = Path(video_path).stem
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, size)
        out_file = out_dir / f"{name_base}_{i}.jpg"
        cv2.imwrite(str(out_file), frame)
        saved += 1
    cap.release()
    return saved


def prepare(src_dir='data/ai_videos', out_dir='data/ai_detector', frames_per_video=8, size=(224,224)):
    src = Path(src_dir)
    out = Path(out_dir)
    for split in ['train', 'val']:
        for cls in ['ai', 'real']:
            src_folder = src / split / cls
            dst_folder = out / split / cls
            dst_folder.mkdir(parents=True, exist_ok=True)
            if not src_folder.exists():
                continue
            files = [f for f in src_folder.glob('*.mp4')]
            print(f"Processing {len(files)} videos from {src_folder} -> {dst_folder}")
            for f in files:
                saved = extract_frames_from_video(f, dst_folder, frames_per_video=frames_per_video, size=size)
                if saved:
                    print(f"  Saved {saved} frames from {f.name}")

    print("Frame extraction complete.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', default='data/ai_videos')
    parser.add_argument('--out_dir', default='data/ai_detector')
    parser.add_argument('--frames_per_video', type=int, default=8)
    args = parser.parse_args()
    prepare(args.src_dir, args.out_dir, frames_per_video=args.frames_per_video)
