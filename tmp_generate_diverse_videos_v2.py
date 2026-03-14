import cv2
import numpy as np
import os
import random
import glob
from pathlib import Path

IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')


def collect_images(directory, limit_per_subdir=None):
    images = []
    if not directory.exists():
        return images

    subdirs = [path for path in directory.iterdir() if path.is_dir()]
    if subdirs:
        for subdir in subdirs:
            subdir_files = []
            for pattern in IMAGE_EXTENSIONS:
                subdir_files.extend(subdir.glob(pattern))
            if limit_per_subdir is not None:
                subdir_files = random.sample(subdir_files, min(limit_per_subdir, len(subdir_files)))
            images.extend(subdir_files)
        return images

    for pattern in IMAGE_EXTENSIONS:
        images.extend(directory.glob(pattern))
    return images

def create_video(src_image_path, out_file, duration_sec=3, fps=15, is_ai=False):
    img = cv2.imread(str(src_image_path)) if src_image_path else None
    if img is None:
        # Fallback noise
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # standardize to 640x480
    img = cv2.resize(img, (640, 480))
    
    # Choose a unique crop part for diversity
    ch, cw = img.shape[:2]
    # random crop if possible
    crop_size = 0.8 # 80% range for variety
    nh, nw = int(ch*crop_size), int(cw*crop_size)
    y1 = random.randint(0, ch - nh)
    x1 = random.randint(0, cw - nw)
    img = img[y1:y1+nh, x1:x1+nw]
    img = cv2.resize(img, (640, 480))
    
    # VideoWriter setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_file), fourcc, fps, (640, 480))
    
    total_frames = int(duration_sec * fps)
    
    # Motion Profile: randomized
    motion_type = random.choice(['pan', 'zoom', 'shake', 'drift'])
    x_drift = random.uniform(-1, 1)
    y_drift = random.uniform(-1, 1)
    zoom_val = 1.0 + random.uniform(-0.02, 0.05)
    
    for i in range(total_frames):
        # Apply transformation
        # Scaling/Zoom
        z_curr = zoom_val ** (i / total_frames)
        dx = x_drift * i
        dy = y_drift * i
        
        M = np.float32([[z_curr, 0, dx], [0, z_curr, dy]])
        frame = cv2.warpAffine(img, M, (640, 480))
        
        if is_ai:
            # AI (Fake) Artifacts:
            # 1. Warp Shift: Simulate temporal inconsistency
            if i % 6 == 0:
                rows, cols = frame.shape[:2]
                map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
                map_x = map_x.astype(np.float32) + 3 * np.sin(map_y / 15.0)
                map_y = map_y.astype(np.float32) + 3 * np.cos(map_x / 15.0)
                frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
            
            # 2. Hue Fluctuation
            if i % 4 == 0:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + random.randint(5, 12)) % 180
                frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                
            # 3. Blocky noise (compression)
            if i % 10 == 0:
                small = cv2.resize(frame, (120, 90))
                frame = cv2.resize(small, (640, 480), cv2.INTER_NEAREST)
        else:
            # REAL video artifacts
            # Subtle Gaussian noise
            noise = np.random.normal(0, 1.2, frame.shape).astype(np.uint8)
            frame = cv2.add(frame, noise)

        out.write(frame)
    
    out.release()

def main():
    base_dir = Path(__file__).resolve().parent
    real_out = base_dir / "data" / "ai_videos" / "train" / "real"
    ai_out = base_dir / "data" / "ai_videos" / "train" / "ai"
    
    # 1. Clean Directories
    import shutil
    shutil.rmtree(real_out, ignore_errors=True)
    shutil.rmtree(ai_out, ignore_errors=True)
    real_out.mkdir(parents=True, exist_ok=True)
    ai_out.mkdir(parents=True, exist_ok=True)
    
    # 2. Select 35 Diverse REAL images
    train_dir = base_dir / "data" / "images" / "train"
    real_images = collect_images(train_dir, limit_per_subdir=2)
    
    # Add one more (random) to reach 35
    if real_images:
        real_images.append(random.choice(real_images))
        
    random.shuffle(real_images)
    print(f"Generating 35 REAL videos with diverse contents...")
    for i, img_path in enumerate(real_images[:35]):
        out_file = real_out / f"real_{i}.mp4"
        create_video(img_path, out_file, duration_sec=random.uniform(2, 4), is_ai=False)
        if i % 10 == 0: print(f"  Real {i}/35")
        
    # 3. Select 50 AI "instances"
    ai_source_dir = base_dir / "data" / "ai_detector" / "train" / "ai"
    ai_sources = collect_images(ai_source_dir)
    
    print(f"Generating 50 AI videos using {len(ai_sources)} diverse sources and unique transformations...")
    for i in range(50):
        # Pick one AI image and use a different transformation seed
        src_path = random.choice(ai_sources) if ai_sources else None
        out_file = ai_out / f"ai_{i}.mp4"
        create_video(src_path, out_file, duration_sec=random.uniform(2, 4), is_ai=True)
        if i % 10 == 0: print(f"  AI {i}/50")

    print("\n✅ Dataset generation successful!")

if __name__ == "__main__":
    main()
