"""
Split a total number of samples evenly across multiple folders (classes).

First renames all files in each source folder with folder name prefix to avoid
name conflicts, then calculates samples per folder (total_samples / number_of_folders),
randomly samples that many images from each folder, and copies ALL images into
a single flat output folder.

Usage (from repo root or from src/):
  py src/tools/split_samples_across_folders.py --source src/tools/potato --total 1000 --output dataset
  py src/tools/split_samples_across_folders.py --source src/tools/potato --total 1000 --output dataset --dry-run
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_image_files(folder: Path):
    """Get all image files in a folder (non-recursive)."""
    if not folder.is_dir():
        return []
    return [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]


def get_class_folders(source_dir: Path):
    """Get all subdirectories in source_dir that contain image files."""
    if not source_dir.is_dir():
        return []
    
    class_folders = []
    for item in source_dir.iterdir():
        if item.is_dir():
            images = get_image_files(item)
            if images:
                class_folders.append(item)
    
    return sorted(class_folders)


def rename_files_in_folder(folder: Path, class_number: int, dry_run: bool = False):
    """
    Rename all image files in a folder by adding a class prefix (c1_, c2_, etc.).
    Skips files that already have the correct prefix.
    
    Args:
        folder: Folder containing images to rename
        class_number: Number for the prefix (1, 2, 3, ...)
        dry_run: If True, only show what would be renamed
    
    Returns:
        Dictionary mapping old paths to new paths
    """
    images = get_image_files(folder)
    if not images:
        return {}
    
    renamed_map = {}
    prefix = f"c{class_number}_"
    
    for img_path in sorted(images):
        # Check if already has the correct prefix
        if img_path.name.startswith(prefix):
            continue
        
        # Create new filename with prefix
        new_name = f"{prefix}{img_path.name}"
        new_path = folder / new_name
        
        # Handle conflicts (shouldn't happen, but just in case)
        if new_path.exists():
            counter = 1
            stem = img_path.stem
            suffix = img_path.suffix
            while new_path.exists():
                new_name = f"{prefix}{stem}_{counter}{suffix}"
                new_path = folder / new_name
                counter += 1
        
        if dry_run:
            print(f"  WOULD RENAME: {img_path.name} -> {new_name}")
        else:
            try:
                img_path.rename(new_path)
                renamed_map[img_path] = new_path
            except Exception as e:
                print(f"  ERROR renaming {img_path.name}: {e}")
    
    return renamed_map


def split_samples_across_folders(
    source_dir: Path,
    total_samples: int,
    output_dir: Path,
    dry_run: bool = False,
    seed: int | None = None
):
    """
    Split total_samples evenly across class folders and sample images.
    
    Args:
        source_dir: Directory containing class subfolders
        total_samples: Total number of samples to collect
        output_dir: Directory where sampled images will be copied
        dry_run: If True, only show what would be done
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
    
    # Get class folders
    class_folders = get_class_folders(source_dir)
    
    if not class_folders:
        raise SystemExit(f"Error: No class folders with images found in '{source_dir}'")
    
    num_folders = len(class_folders)
    samples_per_folder = total_samples // num_folders
    remainder = total_samples % num_folders
    
    print("=== Sample Splitter ===")
    print(f"Source directory: {source_dir.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Total samples requested: {total_samples}")
    print(f"Number of class folders: {num_folders}")
    print(f"Samples per folder: {samples_per_folder}")
    if remainder > 0:
        print(f"Remainder: {remainder} samples will be distributed to first {remainder} folders")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'COPY FILES'}")
    print("-" * 60)
    
    # Step 1: Rename all files in each folder to avoid conflicts
    print("\nStep 1: Renaming files in source folders...")
    print("-" * 60)
    
    folder_rename_maps = {}  # Maps folder -> {old_path: new_path}
    
    for idx, folder in enumerate(class_folders, start=1):
        print(f"\nRenaming files in '{folder.name}' (c{idx}_ prefix):")
        rename_map = rename_files_in_folder(folder, idx, dry_run=dry_run)
        folder_rename_maps[folder] = rename_map
        if not dry_run:
            print(f"  Renamed {len(rename_map)} files")
    
    print("-" * 60)
    
    # Collect images from each folder (after renaming)
    folder_images = {}
    for folder in class_folders:
        images = get_image_files(folder)
        folder_images[folder] = images
        print(f"Folder '{folder.name}': {len(images)} images available")
    
    print("-" * 60)
    
    # Check if we have enough images
    total_available = sum(len(images) for images in folder_images.values())
    if total_available < total_samples:
        print(f"WARNING: Only {total_available} images available, but {total_samples} requested.")
        print(f"Will sample all available images.")
    
    # Create output directory (single flat folder)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nStep 2: Sampling and copying images...")
    print("-" * 60)
    
    # Sample from each folder and collect all selected images
    total_copied = 0
    all_selected = []  # List of (source_path, folder_name) tuples
    
    for idx, class_folder in enumerate(class_folders):
        images = folder_images[class_folder]
        
        # Calculate samples for this folder
        samples_this_folder = samples_per_folder
        if idx < remainder:
            samples_this_folder += 1
        
        # Limit to available images
        samples_this_folder = min(samples_this_folder, len(images))
        
        if samples_this_folder == 0:
            print(f"SKIP '{class_folder.name}': No samples allocated")
            continue
        
        # Randomly sample
        selected = random.sample(images, samples_this_folder)
        
        print(f"\nFolder '{class_folder.name}':")
        print(f"  Sampling {samples_this_folder} from {len(images)} available")
        
        # Add to collection with folder name for conflict resolution
        for img_path in selected:
            all_selected.append((img_path, class_folder.name))
        
        total_copied += samples_this_folder
    
    # Copy all selected images to single output folder
    print(f"\nCopying {total_copied} images to single output folder...")
    copied_count = 0
    
    for img_path, folder_name in all_selected:
        # Files are already renamed with folder prefix, so just use the name directly
        dest_name = img_path.name
        dest_path = output_dir / dest_name
        
        # Handle name conflicts (shouldn't happen since files are renamed, but just in case)
        if dest_path.exists():
            counter = 1
            stem = img_path.stem
            suffix = img_path.suffix
            while dest_path.exists():
                dest_name = f"{stem}_{counter}{suffix}"
                dest_path = output_dir / dest_name
                counter += 1
        
        if dry_run:
            print(f"  WOULD COPY: {img_path.name} -> {dest_name}")
        else:
            try:
                shutil.copy2(img_path, dest_path)
                copied_count += 1
            except Exception as e:
                print(f"  ERROR copying {img_path.name}: {e}")
    
    if not dry_run:
        print(f"\nCopied {copied_count} files to '{output_dir.name}/'")
    
    print("-" * 60)
    print(f"Summary: {total_copied} samples ({samples_per_folder} per folder) copied to single output folder")
    
    if dry_run and total_copied > 0:
        print(f"\nRun without --dry-run to copy these files.")


def main():
    ap = argparse.ArgumentParser(
        description="Split total samples evenly across class folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sample 1000 images evenly across all folders in potato directory
  py src/tools/split_samples_across_folders.py --source src/tools/potato --total 1000 --output dataset
  
  # Preview what would be done
  py src/tools/split_samples_across_folders.py --source src/tools/potato --total 1000 --output dataset --dry-run
  
  # Use a specific random seed for reproducibility
  py src/tools/split_samples_across_folders.py --source src/tools/potato --total 1000 --output dataset --seed 42
        """
    )
    ap.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source directory containing class subfolders with images"
    )
    ap.add_argument(
        "--total",
        type=int,
        required=True,
        help="Total number of samples to collect across all folders"
    )
    ap.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory where all sampled images will be copied (single flat folder)"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without actually copying files"
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling"
    )
    
    args = ap.parse_args()
    
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    
    if not source_dir.exists():
        raise SystemExit(f"Error: Source directory '{source_dir}' does not exist.")
    
    split_samples_across_folders(
        source_dir=source_dir,
        total_samples=args.total,
        output_dir=output_dir,
        dry_run=args.dry_run,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

