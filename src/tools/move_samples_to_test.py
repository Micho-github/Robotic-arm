"""
Move 20% of samples from each class in dataset/train to dataset/test.
Samples are split evenly across subtypes based on prefix (c1, c2, c3, etc.).

For example, if a class has 1000 files and 4 subtypes (c1, c2, c3, c4), then
200 samples (20% of 1000) will be moved, with 50 samples from each subtype.

Usage (from repo root):
  py src/tools/move_samples_to_test.py
  py src/tools/move_samples_to_test.py --dry-run
  py src/tools/move_samples_to_test.py --percentage 0.2 --prefix c
  py src/tools/move_samples_to_test.py --samples 200  # Use fixed number instead
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_image_files(folder: Path):
    """Get all image files in a folder (non-recursive)."""
    if not folder.is_dir():
        return []
    return [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]


def extract_subtype_prefix(filename: str, prefix: str = "c") -> str | None:
    """
    Extract subtype prefix from filename.
    
    Examples:
        c1_10_100.jpg -> c1
        c2_r_100_100.jpg -> c2
        c3_50_100.jpg -> c3
    
    Args:
        filename: Name of the file
        prefix: The prefix to look for (default: "c")
    
    Returns:
        Subtype prefix (e.g., "c1", "c2") or None if not found
    """
    # Pattern: prefix followed by digits (e.g., c1, c2, c3)
    pattern = rf"^{re.escape(prefix)}(\d+)"
    match = re.match(pattern, filename)
    if match:
        return f"{prefix}{match.group(1)}"
    return None


def group_files_by_subtype(files: list[Path], prefix: str = "c") -> dict[str, list[Path]]:
    """
    Group files by their subtype prefix.
    
    Args:
        files: List of file paths
        prefix: Prefix to look for (default: "c")
    
    Returns:
        Dictionary mapping subtype (e.g., "c1") to list of files
    """
    grouped = defaultdict(list)
    for file_path in files:
        subtype = extract_subtype_prefix(file_path.name, prefix)
        if subtype:
            grouped[subtype].append(file_path)
    return dict(grouped)


def move_samples_to_test(
    train_dir: Path,
    test_dir: Path,
    percentage: float | None = 0.2,
    samples_per_class: int | None = None,
    prefix: str = "c",
    dry_run: bool = False,
    seed: int | None = None
):
    """
    Move samples from train to test, split evenly across subtypes.
    
    Args:
        train_dir: Directory containing class subfolders with training data
        test_dir: Directory where test samples will be moved
        percentage: Percentage of samples to move per class (default: 0.2 for 20%)
        samples_per_class: Fixed number of samples to move per class (overrides percentage if set)
        prefix: Prefix to identify subtypes (default: "c")
        dry_run: If True, only show what would be moved
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
    
    if not train_dir.exists():
        raise SystemExit(f"Error: Train directory '{train_dir}' does not exist.")
    
    # Get all class folders
    class_folders = []
    for item in train_dir.iterdir():
        if item.is_dir():
            images = get_image_files(item)
            if images:
                class_folders.append(item)
    
    if not class_folders:
        raise SystemExit(f"Error: No class folders with images found in '{train_dir}'")
    
    class_folders = sorted(class_folders)
    
    print("=== Move Samples to Test ===")
    print(f"Train directory: {train_dir.resolve()}")
    print(f"Test directory: {test_dir.resolve()}")
    if samples_per_class is not None:
        print(f"Fixed samples per class: {samples_per_class}")
    else:
        print(f"Percentage per class: {percentage * 100:.1f}%")
    print(f"Subtype prefix: {prefix}")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'MOVE FILES'}")
    print(f"Number of classes: {len(class_folders)}")
    print("-" * 60)
    
    total_moved = 0
    
    for class_folder in class_folders:
        print(f"\nProcessing class: '{class_folder.name}'")
        print("-" * 60)
        
        # Get all image files in this class
        all_files = get_image_files(class_folder)
        total_files = len(all_files)
        print(f"Total files in train: {total_files}")
        
        # Calculate samples to move for this class
        if samples_per_class is not None:
            samples_to_move = samples_per_class
        else:
            samples_to_move = int(total_files * percentage)
        
        print(f"Samples to move: {samples_to_move} ({samples_to_move/total_files*100:.1f}%)")
        
        # Group files by subtype
        subtype_groups = group_files_by_subtype(all_files, prefix)
        
        if not subtype_groups:
            print(f"  WARNING: No files with prefix '{prefix}' found. Skipping.")
            continue
        
        num_subtypes = len(subtype_groups)
        samples_per_subtype = samples_to_move // num_subtypes
        remainder = samples_to_move % num_subtypes
        
        print(f"Found {num_subtypes} subtypes: {sorted(subtype_groups.keys())}")
        print(f"Samples per subtype: {samples_per_subtype}")
        if remainder > 0:
            print(f"Remainder: {remainder} extra samples will be distributed to first {remainder} subtypes")
        
        # Create test directory for this class
        test_class_dir = test_dir / class_folder.name
        if not dry_run:
            test_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample from each subtype
        selected_files = []
        subtype_names = sorted(subtype_groups.keys())
        
        for idx, subtype in enumerate(subtype_names):
            files_in_subtype = subtype_groups[subtype]
            
            # Calculate samples for this subtype
            samples_this_subtype = samples_per_subtype
            if idx < remainder:
                samples_this_subtype += 1
            
            # Limit to available files
            samples_this_subtype = min(samples_this_subtype, len(files_in_subtype))
            
            if samples_this_subtype == 0:
                print(f"  {subtype}: SKIP (no samples allocated)")
                continue
            
            # Randomly sample
            selected = random.sample(files_in_subtype, samples_this_subtype)
            selected_files.extend(selected)
            
            print(f"  {subtype}: {len(files_in_subtype)} available, selecting {samples_this_subtype}")
        
        # Move selected files
        print(f"\nMoving {len(selected_files)} files to test...")
        moved_count = 0
        
        for file_path in selected_files:
            dest_path = test_class_dir / file_path.name
            
            # Check if destination already exists (shouldn't happen, but just in case)
            if dest_path.exists():
                print(f"  WARNING: {file_path.name} already exists in test. Skipping.")
                continue
            
            if dry_run:
                print(f"  WOULD MOVE: {file_path.name}")
            else:
                try:
                    shutil.move(str(file_path), str(dest_path))
                    moved_count += 1
                except Exception as e:
                    print(f"  ERROR moving {file_path.name}: {e}")
        
        if not dry_run:
            print(f"Moved {moved_count} files from '{class_folder.name}' to test")
            remaining = len(get_image_files(class_folder))
            print(f"Remaining in train: {remaining} files")
        
        total_moved += moved_count
    
    print("\n" + "=" * 60)
    print(f"Summary: {total_moved} files moved to test")
    
    if dry_run and total_moved > 0:
        print(f"\nRun without --dry-run to apply these changes.")


def main():
    ap = argparse.ArgumentParser(
        description="Move samples from train to test, split evenly across subtypes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Move 20%% of samples per class (default)
  py src/tools/move_samples_to_test.py
  
  # Preview what would be moved
  py src/tools/move_samples_to_test.py --dry-run
  
  # Move 15%% of samples per class
  py src/tools/move_samples_to_test.py --percentage 0.15
  
  # Move fixed 200 samples per class (overrides percentage)
  py src/tools/move_samples_to_test.py --samples 200
  
  # Use a different prefix (e.g., 'fg' instead of 'c')
  py src/tools/move_samples_to_test.py --prefix fg
  
  # Use a specific random seed for reproducibility
  py src/tools/move_samples_to_test.py --seed 42
        """
    )
    ap.add_argument(
        "--train-dir",
        type=str,
        default="src/models/conveyor/dataset/train",
        help="Directory containing class subfolders with training data (default: src/models/conveyor/dataset/train)"
    )
    ap.add_argument(
        "--test-dir",
        type=str,
        default="src/models/conveyor/dataset/test",
        help="Directory where test samples will be moved (default: src/models/conveyor/dataset/test)"
    )
    ap.add_argument(
        "--percentage",
        type=float,
        default=0.2,
        help="Percentage of samples to move per class (default: 0.2 for 20%%)"
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Fixed number of samples to move per class (overrides --percentage if set)"
    )
    ap.add_argument(
        "--prefix",
        type=str,
        default="c",
        help="Prefix to identify subtypes (default: 'c')"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without actually moving files"
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling"
    )
    
    args = ap.parse_args()
    
    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    
    move_samples_to_test(
        train_dir=train_dir,
        test_dir=test_dir,
        percentage=args.percentage,
        samples_per_class=args.samples,
        prefix=args.prefix,
        dry_run=args.dry_run,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

