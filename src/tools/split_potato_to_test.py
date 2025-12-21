"""
Move 20% of samples from dataset/train/potato to dataset/test/potato.
Samples are split evenly across subtypes based on prefix (c1_, c2, c3, c4).

For example, if potato has 1000 files and 4 subtypes (c1_, c2, c3, c4), then
200 samples (20% of 1000) will be moved, with 50 samples from each subtype.

Usage (from repo root):
  py src/tools/split_potato_to_test.py
  py src/tools/split_potato_to_test.py --dry-run
  py src/tools/split_potato_to_test.py --percentage 0.2
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


def extract_subtype_prefix(filename: str) -> str | None:
    """
    Extract subtype prefix from filename.
    
    Handles both c1_ (with underscore) and c2, c3, c4 (without underscore).
    
    Examples:
        c1_10_100.jpg -> c1
        c2_r_100_100.jpg -> c2
        c3_50_100.jpg -> c3
        c4_75_100.jpg -> c4
    
    Args:
        filename: Name of the file
    
    Returns:
        Subtype prefix (e.g., "c1", "c2") or None if not found
    """
    # Pattern 1: c1_ (with underscore)
    if filename.startswith("c1_"):
        return "c1"
    
    # Pattern 2: c2, c3, c4 (without underscore, followed by non-digit or end)
    pattern = r"^c([2-4])(?![0-9])"
    match = re.match(pattern, filename)
    if match:
        return f"c{match.group(1)}"
    
    return None


def group_files_by_subtype(files: list[Path]) -> dict[str, list[Path]]:
    """
    Group files by their subtype prefix.
    
    Args:
        files: List of file paths
    
    Returns:
        Dictionary mapping subtype (e.g., "c1", "c2") to list of files
    """
    grouped = defaultdict(list)
    for file_path in files:
        subtype = extract_subtype_prefix(file_path.name)
        if subtype:
            grouped[subtype].append(file_path)
    return dict(grouped)


def split_potato_to_test(
    train_potato_dir: Path,
    test_potato_dir: Path,
    percentage: float = 0.2,
    dry_run: bool = False,
    seed: int | None = None
):
    """
    Move samples from train/potato to test/potato, split evenly across subtypes.
    
    Args:
        train_potato_dir: Directory containing potato training data
        test_potato_dir: Directory where test samples will be moved
        percentage: Percentage of samples to move (default: 0.2 for 20%)
        dry_run: If True, only show what would be moved
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
    
    if not train_potato_dir.exists():
        raise SystemExit(f"Error: Train potato directory '{train_potato_dir}' does not exist.")
    
    print("=== Split Potato to Test ===")
    print(f"Train directory: {train_potato_dir.resolve()}")
    print(f"Test directory: {test_potato_dir.resolve()}")
    print(f"Percentage to move: {percentage * 100:.1f}%")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'MOVE FILES'}")
    print("-" * 60)
    
    # Get all image files in potato folder
    all_files = get_image_files(train_potato_dir)
    total_files = len(all_files)
    
    if total_files == 0:
        raise SystemExit(f"Error: No image files found in '{train_potato_dir}'")
    
    print(f"Total files in train: {total_files}")
    
    # Calculate samples to move
    samples_to_move = int(total_files * percentage)
    print(f"Samples to move: {samples_to_move} ({samples_to_move/total_files*100:.1f}%)")
    
    # Group files by subtype
    subtype_groups = group_files_by_subtype(all_files)
    
    if not subtype_groups:
        raise SystemExit(f"Error: No files with valid subtype prefixes (c1_, c2, c3, c4) found.")
    
    num_subtypes = len(subtype_groups)
    samples_per_subtype = samples_to_move // num_subtypes
    remainder = samples_to_move % num_subtypes
    
    print(f"\nFound {num_subtypes} subtypes: {sorted(subtype_groups.keys())}")
    print(f"Samples per subtype: {samples_per_subtype}")
    if remainder > 0:
        print(f"Remainder: {remainder} extra samples will be distributed to first {remainder} subtypes")
    
    # Create test directory
    if not dry_run:
        test_potato_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample from each subtype
    selected_files = []
    subtype_names = sorted(subtype_groups.keys())
    
    print("\nSampling from each subtype:")
    print("-" * 60)
    
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
    print("-" * 60)
    moved_count = 0
    
    for file_path in selected_files:
        dest_path = test_potato_dir / file_path.name
        
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
        remaining = len(get_image_files(train_potato_dir))
        print(f"\nMoved {moved_count} files to test")
        print(f"Remaining in train: {remaining} files")
    
    print("\n" + "=" * 60)
    print(f"Summary: {moved_count} files moved to test")
    
    if dry_run and moved_count > 0:
        print(f"\nRun without --dry-run to apply these changes.")


def main():
    ap = argparse.ArgumentParser(
        description="Move samples from train/potato to test/potato, split evenly across subtypes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Move 20%% of samples (default)
  py src/tools/split_potato_to_test.py
  
  # Preview what would be moved
  py src/tools/split_potato_to_test.py --dry-run
  
  # Move 15%% of samples
  py src/tools/split_potato_to_test.py --percentage 0.15
  
  # Use a specific random seed for reproducibility
  py src/tools/split_potato_to_test.py --seed 42
        """
    )
    ap.add_argument(
        "--train-dir",
        type=str,
        default="src/models/conveyor/dataset/train/potato",
        help="Directory containing potato training data (default: src/models/conveyor/dataset/train/potato)"
    )
    ap.add_argument(
        "--test-dir",
        type=str,
        default="src/models/conveyor/dataset/test/potato",
        help="Directory where test samples will be moved (default: src/models/conveyor/dataset/test/potato)"
    )
    ap.add_argument(
        "--percentage",
        type=float,
        default=0.2,
        help="Percentage of samples to move (default: 0.2 for 20%%)"
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
    
    train_potato_dir = Path(args.train_dir)
    test_potato_dir = Path(args.test_dir)
    
    split_potato_to_test(
        train_potato_dir=train_potato_dir,
        test_potato_dir=test_potato_dir,
        percentage=args.percentage,
        dry_run=args.dry_run,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

