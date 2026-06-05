"""
Download COCO train2017 and/or val2017 images for Thai captioning.

Usage:
    python download_coco.py --split val    # ~1 GB,  ~5 min  (download eval set first)
    python download_coco.py --split train  # ~18 GB, ~45 min  (run in background)
    python download_coco.py --split both
    python download_coco.py --verify-only  # count existing images, no download
"""

import argparse
import os
import subprocess
import zipfile
from pathlib import Path

BASE_DIR = "/teamspace/studios/this_studio"
COCO_DIR = os.path.join(BASE_DIR, "coco")

URLS = {
    "train": "http://images.cocodataset.org/zips/train2017.zip",
    "val":   "http://images.cocodataset.org/zips/val2017.zip",
}
EXPECTED_COUNTS = {
    "train": 118287,
    "val":   5000,
}


def count_images(split: str) -> int:
    target = os.path.join(COCO_DIR, f"{split}2017")
    if not os.path.isdir(target):
        return 0
    return sum(1 for f in os.scandir(target) if f.name.endswith(".jpg"))


def download_and_extract(split: str) -> None:
    os.makedirs(COCO_DIR, exist_ok=True)
    zip_path = os.path.join(COCO_DIR, f"{split}2017.zip")
    url = URLS[split]

    existing = count_images(split)
    expected = EXPECTED_COUNTS[split]

    if existing >= expected:
        print(f"[{split}] Already complete: {existing}/{expected} images found.")
        return

    print(f"[{split}] Downloading {url} ...")
    # -c: resume if interrupted; -P: output directory
    result = subprocess.run(
        ["wget", "-c", "-P", COCO_DIR, url],
        check=True,
    )

    print(f"[{split}] Extracting {zip_path} to {COCO_DIR} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(COCO_DIR)

    # Remove zip to save disk space
    os.remove(zip_path)
    print(f"[{split}] Removed {zip_path}")

    final_count = count_images(split)
    status = "OK" if final_count >= expected else "WARNING: incomplete"
    print(f"[{split}] {status} — {final_count}/{expected} images present at {COCO_DIR}/{split}2017/")


def verify_only() -> None:
    for split in ("train", "val"):
        count = count_images(split)
        expected = EXPECTED_COUNTS[split]
        pct = count / expected * 100 if expected else 0
        status = "complete" if count >= expected else f"missing {expected - count}"
        print(f"  {split}2017: {count}/{expected} images ({pct:.1f}%) — {status}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["train", "val", "both"],
        default="both",
        help="Which COCO split to download",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only count existing images, do not download",
    )
    args = parser.parse_args()

    if args.verify_only:
        print("COCO image counts:")
        verify_only()
        return

    splits = ["train", "val"] if args.split == "both" else [args.split]
    for split in splits:
        download_and_extract(split)

    print("\nFinal verification:")
    verify_only()


if __name__ == "__main__":
    main()
