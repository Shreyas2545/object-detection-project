from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment class folders until they reach a minimum image count.")
    parser.add_argument("--root", default="data/images/train", help="Root directory containing one folder per class.")
    parser.add_argument("--target", type=int, default=2500, help="Minimum number of images required per class.")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Optional class folder names to augment (e.g. --classes headphone glasses).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible augmentation.")
    return parser.parse_args()


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        path for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def random_crop_resize(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    crop_ratio = rng.uniform(0.84, 0.98)
    crop_width = max(1, int(width * crop_ratio))
    crop_height = max(1, int(height * crop_ratio))
    max_left = max(0, width - crop_width)
    max_top = max(0, height - crop_height)
    left = rng.randint(0, max_left) if max_left else 0
    top = rng.randint(0, max_top) if max_top else 0
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), Image.Resampling.BILINEAR)


def augment_image(image: Image.Image, rng: random.Random) -> Image.Image:
    result = image.convert("RGB")

    if rng.random() < 0.5:
        result = ImageOps.mirror(result)

    if rng.random() < 0.2:
        result = ImageOps.flip(result)

    result = random_crop_resize(result, rng)

    rotation = rng.uniform(-18, 18)
    result = result.rotate(rotation, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255))

    result = ImageEnhance.Brightness(result).enhance(rng.uniform(0.85, 1.18))
    result = ImageEnhance.Contrast(result).enhance(rng.uniform(0.85, 1.2))
    result = ImageEnhance.Color(result).enhance(rng.uniform(0.88, 1.15))
    if rng.random() < 0.15:
        result = result.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.0)))

    return result


def next_augmented_path(class_dir: Path, class_name: str, index: int) -> Path:
    return class_dir / f"aug_{class_name}_{index:05d}.jpg"


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_existing_hashes(class_dir: Path) -> set[str]:
    hashes: set[str] = set()
    for image_path in list_images(class_dir):
        try:
            hashes.add(file_sha256(image_path))
        except OSError:
            continue
    return hashes


def augment_class(class_dir: Path, target: int, rng: random.Random) -> tuple[int, int]:
    images = list_images(class_dir)
    original_count = len(images)

    if original_count == 0 or original_count >= target:
        return original_count, original_count

    class_name = class_dir.name.lower()
    existing_augmented = len(list(class_dir.glob(f"aug_{class_name}_*.jpg")))
    needed = target - original_count
    existing_hashes = collect_existing_hashes(class_dir)

    source_images = images[:]
    rng.shuffle(source_images)

    created = 0
    attempts = 0
    max_attempts = max(needed * 15, 100)

    while created < needed and attempts < max_attempts:
        source_path = source_images[(existing_augmented + created + attempts) % len(source_images)]
        with Image.open(source_path) as image:
            augmented = augment_image(image, rng)

        output_index = existing_augmented + created + 1
        output_path = next_augmented_path(class_dir, class_name, output_index)
        quality = rng.randint(88, 94)
        augmented.save(output_path, format="JPEG", quality=quality)

        digest = file_sha256(output_path)
        if digest in existing_hashes:
            output_path.unlink(missing_ok=True)
            attempts += 1
            continue

        existing_hashes.add(digest)
        created += 1
        attempts += 1

    if created < needed:
        print(f"Warning: {class_dir.name} reached {original_count + created} images (target {target}) due to duplicate filtering")

    return original_count, original_count + created


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rng = random.Random(args.seed)

    if not root.exists():
        raise SystemExit(f"Dataset root not found: {root}")

    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if args.classes:
        class_filter = {name.strip().lower() for name in args.classes}
        class_dirs = [path for path in class_dirs if path.name.lower() in class_filter]

    if not class_dirs:
        raise SystemExit(f"No class folders found under: {root}")

    print(f"Augmenting classes under {root} to at least {args.target} images each")
    for class_dir in class_dirs:
        before, after = augment_class(class_dir, args.target, rng)
        delta = after - before
        status = f"+{delta}" if delta > 0 else "unchanged"
        print(f"{class_dir.name}: {before} -> {after} ({status})")


if __name__ == "__main__":
    main()