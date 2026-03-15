from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download seed images for object classes.")
    parser.add_argument("--root", default="data/images/train", help="Train root directory.")
    parser.add_argument("--per-class", type=int, default=450, help="Target seed images per class.")
    parser.add_argument("--classes", nargs="+", required=True, help="Class names to download.")
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def existing_hashes(class_dir: Path) -> set[str]:
    hashes: set[str] = set()
    for path in class_dir.glob("*"):
        if not path.is_file():
            continue
        try:
            with path.open("rb") as handle:
                hashes.add(sha256_bytes(handle.read()))
        except OSError:
            continue
    return hashes


def normalized_query(name: str) -> str:
    mapping = {
        "remote": "tv remote control in hand",
        "charger": "phone charger cable adapter",
        "umbrella": "open umbrella object street",
    }
    return mapping.get(name.lower(), name)


def download_class(class_name: str, root: Path, per_class: int) -> tuple[int, int]:
    from icrawler.builtin import BingImageCrawler

    class_dir = root / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    before = len([p for p in class_dir.iterdir() if p.is_file()])
    hashes = existing_hashes(class_dir)

    if before >= per_class:
        return before, before

    needed = per_class - before
    tmp_dir = class_dir / "_tmp_download"
    if tmp_dir.exists():
        for p in tmp_dir.glob("**/*"):
            if p.is_file():
                p.unlink(missing_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    query = normalized_query(class_name)
    max_num = max(needed * 3, 300)

    crawler = BingImageCrawler(storage={"root_dir": str(tmp_dir)})
    crawler.crawl(keyword=query, max_num=max_num)

    moved = 0
    next_idx = before + 1

    for src in sorted(tmp_dir.glob("**/*")):
        if not src.is_file():
            continue
        if moved >= needed:
            break

        try:
            with Image.open(src) as image:
                image = image.convert("RGB")
                image.thumbnail((1024, 1024))
                out_path = class_dir / f"seed_{class_name}_{next_idx:05d}.jpg"
                image.save(out_path, format="JPEG", quality=90)

            with out_path.open("rb") as handle:
                digest = sha256_bytes(handle.read())

            if digest in hashes:
                out_path.unlink(missing_ok=True)
                continue

            hashes.add(digest)
            moved += 1
            next_idx += 1
        except Exception:
            continue

    for p in sorted(tmp_dir.glob("**/*"), reverse=True):
        if p.is_file():
            p.unlink(missing_ok=True)
        elif p.is_dir():
            try:
                p.rmdir()
            except OSError:
                pass
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    after = len([p for p in class_dir.iterdir() if p.is_file()])
    return before, after


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    for class_name in args.classes:
        before, after = download_class(class_name, root, args.per_class)
        print(f"{class_name}: {before} -> {after} (+{after - before})")


if __name__ == "__main__":
    main()
