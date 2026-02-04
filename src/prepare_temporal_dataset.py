import os
import random
import cv2
import numpy as np
from pathlib import Path

"""
Simple dataset preparation for temporal model quick experiments.
- Creates data/ai_videos/train/{ai,real} and data/ai_videos/val/{ai,real}
- Synthesizes 'real' videos by panning/zooming across images from data/images/train/
- Synthesizes 'ai' videos by augmenting frames from src/ai-video-testing.mp4 (if present)

This is a quick utility for local experiments, not for production dataset curation.
"""

OUT_DIR = Path('data/ai_videos')
TRAIN_AI = OUT_DIR / 'train' / 'ai'
TRAIN_REAL = OUT_DIR / 'train' / 'real'
VAL_AI = OUT_DIR / 'val' / 'ai'
VAL_REAL = OUT_DIR / 'val' / 'real'

def ensure_dirs():
    for d in [TRAIN_AI, TRAIN_REAL, VAL_AI, VAL_REAL]:
        d.mkdir(parents=True, exist_ok=True)


def synthesize_real_videos(num=6, frames=30, out_dir=TRAIN_REAL):
    # Use images from data/images/train categories
    imgs = []
    base = Path('data/images/train')
    if base.exists():
        for sub in base.iterdir():
            if sub.is_dir():
                for img in sub.glob('*.*'):
                    imgs.append(str(img))
    if not imgs:
        # fallback: create plain moving gradient videos
        for i in range(num):
            out = out_dir / f'real_{i}.mp4'
            create_gradient_video(str(out), frames=frames)
        return

    random.shuffle(imgs)
    chosen = imgs[:max(1, min(len(imgs), num))]

    for i, img_path in enumerate(chosen):
        out = out_dir / f'real_{i}.mp4'
        create_pan_video(img_path, str(out), frames=frames)


def create_pan_video(image_path, out_path, frames=30, w=320, h=240, fps=15):
    img = cv2.imread(image_path)
    if img is None:
        return
    ih, iw = img.shape[:2]
    # resize preserving aspect
    scale = max(w/iw, h/ih)
    img_resized = cv2.resize(img, (int(iw*scale), int(ih*scale)))

    # create panning by cropping window moving across
    cx = img_resized.shape[1] - w
    cy = img_resized.shape[0] - h
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w,h))

    for f in range(frames):
        t = f / max(1, frames-1)
        x = int(cx * t)
        y = int(cy * (1 - t))
        crop = img_resized[y:y+h, x:x+w]
        writer.write(crop)
    writer.release()


def create_gradient_video(out_path, frames=30, w=320, h=240, fps=15):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w,h))
    for i in range(frames):
        a = np.linspace(0, 255, w, dtype=np.uint8)
        row = np.tile(a, (h,1))
        frame = np.stack([row, np.roll(row, i, axis=1), np.flip(row, axis=1)], axis=2)
        writer.write(frame)
    writer.release()


def synthesize_ai_videos(num=6, frames=30, out_dir=TRAIN_AI):
    # If sample AI video exists, create augmented versions
    sample = Path('src') / 'ai-video-testing.mp4'
    if sample.exists():
        for i in range(num):
            out = out_dir / f'ai_aug_{i}.mp4'
            augment_video(str(sample), str(out), frames=frames, mode=i%3)
    else:
        # fallback: create noisy videos
        for i in range(num):
            out = out_dir / f'ai_{i}.mp4'
            create_noisy_video(str(out), frames=frames)


def augment_video(in_path, out_path, frames=30, mode=0, w=320, h=240, fps=15):
    cap = cv2.VideoCapture(in_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        create_noisy_video(out_path, frames)
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w,h))

    indices = np.linspace(0, total-1, frames, dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            # pad with black
            frame = np.zeros((h,w,3), dtype=np.uint8)
        frame = cv2.resize(frame, (w,h))
        if mode == 0:
            # add subtle jitter
            M = np.float32([[1,0,random.uniform(-1,1)],[0,1,random.uniform(-1,1)]])
            frame = cv2.warpAffine(frame, M, (w,h))
        elif mode == 1:
            # color distort
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame = frame.astype(np.float32)
            frame[...,2] = np.clip(frame[...,2] * random.uniform(0.8,1.2), 0,255)
            frame = frame.astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        else:
            # add slight blur
            frame = cv2.GaussianBlur(frame, (5,5), sigmaX=1.5)
        writer.write(frame)
    writer.release()
    cap.release()


def create_noisy_video(out_path, frames=30, w=320, h=240, fps=15):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w,h))
    for i in range(frames):
        frame = np.random.randint(0,256,(h,w,3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def split_to_val(train_dir, val_dir, count=2):
    files = list(train_dir.glob('*.mp4'))
    if len(files) <= count:
        return
    moved = 0
    for f in files[-count:]:
        f.rename(val_dir / f.name)
        moved += 1


def main():
    ensure_dirs()
    synthesize_real_videos(num=6, frames=30, out_dir=TRAIN_REAL)
    synthesize_ai_videos(num=6, frames=30, out_dir=TRAIN_AI)
    # move some to val
    split_to_val(TRAIN_REAL, VAL_REAL, count=2)
    split_to_val(TRAIN_AI, VAL_AI, count=2)
    print('Dataset prepared under data/ai_videos')

if __name__ == '__main__':
    main()
