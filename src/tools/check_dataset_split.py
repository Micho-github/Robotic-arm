"""
Sanity-check a train/val ImageFolder dataset for leakage.

Checks:
- filename overlap between train and val (per class and globally)
- exact-content overlap using SHA256 hashing (more reliable)

Usage (from repo root or from src/):
  py src/tools/check_dataset_split.py
  py src/tools/check_dataset_split.py --dataset-dir src/models/vision/dataset
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from collections import defaultdict


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def sha256_file(path: Path, chunk_size=1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=str, default="src/models/vision/dataset", help="Dataset root containing train/ and val/")
    ap.add_argument("--max-report", type=int, default=20, help="Max duplicate examples to print")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"

    if not train_dir.is_dir() or not val_dir.is_dir():
        raise SystemExit(f"Expected '{train_dir}' and '{val_dir}' to exist.")

    # Collect per-split files
    train_files = list(iter_images(train_dir))
    val_files = list(iter_images(val_dir))

    print("=== Dataset Split Check ===")
    print(f"Dataset: {dataset_dir.resolve()}")
    print(f"Train images: {len(train_files)}")
    print(f"Val images:   {len(val_files)}")

    # Filename overlap:
    # - Global filename overlap is often meaningless because many datasets reuse the same numbering
    #   per class (e.g., every class has "1_100.jpg"). What matters is overlap within the SAME class.

    def class_of(split_root: Path, p: Path) -> str:
        rel = p.relative_to(split_root)
        return rel.parts[0] if rel.parts else "?"

    train_global_names = {p.name for p in train_files}
    val_global_names = {p.name for p in val_files}
    overlap_names_global = sorted(train_global_names & val_global_names)
    print("\n--- Filename overlap (global; often expected) ---")
    print(f"Overlap count: {len(overlap_names_global)}")
    if overlap_names_global:
        for name in overlap_names_global[: args.max_report]:
            print(f"  - {name}")
        if len(overlap_names_global) > args.max_report:
            print(f"  ... (+{len(overlap_names_global) - args.max_report} more)")

    train_keys = {(class_of(train_dir, p), p.name) for p in train_files}
    val_keys = {(class_of(val_dir, p), p.name) for p in val_files}
    overlap_keys = sorted(train_keys & val_keys)

    print("\n--- Filename overlap (same class + filename) ---")
    print(f"Overlap count: {len(overlap_keys)}")
    if overlap_keys:
        for cls, name in overlap_keys[: args.max_report]:
            print(f"  - {cls}/{name}")
        if len(overlap_keys) > args.max_report:
            print(f"  ... (+{len(overlap_keys) - args.max_report} more)")

    # Hash overlap (exact duplicates)
    print("\nHashing train images (SHA256)...")
    train_hash_to_paths = defaultdict(list)
    for p in train_files:
        train_hash_to_paths[sha256_file(p)].append(p)

    print("Hashing val images (SHA256)...")
    val_hash_to_paths = defaultdict(list)
    for p in val_files:
        val_hash_to_paths[sha256_file(p)].append(p)

    dup_hashes = sorted(set(train_hash_to_paths.keys()) & set(val_hash_to_paths.keys()))
    print("\n--- Exact content overlap (SHA256) ---")
    print(f"Duplicate-content hashes across train/val: {len(dup_hashes)}")

    shown = 0
    for h in dup_hashes:
        if shown >= args.max_report:
            break
        t_paths = train_hash_to_paths[h]
        v_paths = val_hash_to_paths[h]
        print(f"\nHash: {h[:16]}...  train={len(t_paths)} val={len(v_paths)}")
        for tp in t_paths[:2]:
            print(f"  train: {tp}")
        for vp in v_paths[:2]:
            print(f"  val:   {vp}")
        shown += 1

    if len(dup_hashes) > args.max_report:
        print(f"\n... (+{len(dup_hashes) - args.max_report} more duplicate hashes)")

    if not overlap_keys and not dup_hashes:
        print("\nOK: No obvious leakage detected (no same-class filename overlap, no exact duplicate content).")
    else:
        print("\nWARNING: Possible leakage detected. Remove duplicates from either train or val for a fair test.")


if __name__ == "__main__":
    main()


