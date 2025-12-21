"""
Rename files in dataset/train/potato by replacing old prefixes with new c1-c4 prefixes.

Mapping:
  - red_potato_red* → c1_*
  - washed_potato* → c2*
  - sweet_potato* → c3*
  - white_potato* → c4*

Usage (from repo root):
  py src/tools/rename_potato_prefixes.py
  py src/tools/rename_potato_prefixes.py --dry-run
  py src/tools/rename_potato_prefixes.py --folder src/models/conveyor/dataset/train/potato
"""

from __future__ import annotations

import argparse
from pathlib import Path


# Mapping of old prefixes to new prefixes
PREFIX_MAPPING = {
    "red_potato_red": "c1_",
    "washed_potato": "c2",
    "sweet_potato": "c3",
    "white_potato": "c4",
}


def rename_potato_files(folder: Path, dry_run: bool = False):
    """
    Rename files in the potato folder by replacing old prefixes with new c1-c4 prefixes.
    
    Args:
        folder: Directory containing potato files to rename
        dry_run: If True, only print what would be renamed without actually renaming
    """
    if not folder.is_dir():
        raise SystemExit(f"Error: '{folder}' is not a directory.")
    
    # Get all files in the folder
    files = [f for f in folder.iterdir() if f.is_file()]
    
    if not files:
        print(f"No files found in '{folder}'")
        return
    
    print(f"Scanning folder: {folder.resolve()}")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'RENAME'}")
    print("-" * 60)
    
    renamed_count = 0
    skipped_count = 0
    prefix_counts = {old: 0 for old in PREFIX_MAPPING.keys()}
    c1_to_c1_underscore_count = 0
    
    for file_path in sorted(files):
        filename = file_path.name
        
        # First, handle files that already have "c1" prefix but not "c1_"
        if filename.startswith("c1") and not filename.startswith("c1_"):
            new_filename = filename.replace("c1", "c1_", 1)  # Replace only first occurrence
            new_path = file_path.parent / new_filename
            
            if new_path.exists():
                print(f"SKIP (target exists): {filename} -> {new_filename}")
                skipped_count += 1
                continue
            
            if dry_run:
                print(f"WOULD RENAME: {filename} -> {new_filename}")
            else:
                try:
                    file_path.rename(new_path)
                    print(f"RENAMED: {filename} -> {new_filename}")
                    renamed_count += 1
                    c1_to_c1_underscore_count += 1
                except Exception as e:
                    print(f"ERROR renaming {filename}: {e}")
                    skipped_count += 1
            continue
        
        matched_prefix = None
        new_prefix = None
        
        # Check each old prefix to see if filename starts with it
        for old_prefix, new_prefix_value in PREFIX_MAPPING.items():
            if filename.startswith(old_prefix):
                matched_prefix = old_prefix
                new_prefix = new_prefix_value
                break
        
        # Skip if no matching prefix found
        if matched_prefix is None:
            skipped_count += 1
            continue
        
        # Check if already has the new prefix
        if filename.startswith(new_prefix):
            print(f"SKIP (already has {new_prefix} prefix): {filename}")
            skipped_count += 1
            continue
        
        # Create new filename by replacing the old prefix with new prefix
        new_filename = filename.replace(matched_prefix, new_prefix, 1)  # Replace only first occurrence
        new_path = file_path.parent / new_filename
        
        # Check if target already exists
        if new_path.exists():
            print(f"SKIP (target exists): {filename} -> {new_filename}")
            skipped_count += 1
            continue
        
        prefix_counts[matched_prefix] += 1
        
        if dry_run:
            print(f"WOULD RENAME: {filename} -> {new_filename}")
        else:
            try:
                file_path.rename(new_path)
                print(f"RENAMED: {filename} -> {new_filename}")
                renamed_count += 1
            except Exception as e:
                print(f"ERROR renaming {filename}: {e}")
                skipped_count += 1
    
    print("-" * 60)
    print("Summary by prefix:")
    if c1_to_c1_underscore_count > 0:
        print(f"  c1 -> c1_: {c1_to_c1_underscore_count} files")
    for old_prefix, new_prefix in PREFIX_MAPPING.items():
        count = prefix_counts[old_prefix]
        if count > 0:
            print(f"  {old_prefix} -> {new_prefix}: {count} files")
    
    print("-" * 60)
    print(f"Total: {renamed_count} renamed, {skipped_count} skipped")
    
    if dry_run and renamed_count > 0:
        print(f"\nRun without --dry-run to apply these changes.")


def main():
    ap = argparse.ArgumentParser(
        description="Rename potato files by replacing old prefixes with c1-c4 prefixes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rename files in default potato folder
  py src/tools/rename_potato_prefixes.py
  
  # Preview what would be renamed
  py src/tools/rename_potato_prefixes.py --dry-run
  
  # Rename files in a different folder
  py src/tools/rename_potato_prefixes.py --folder path/to/potato/folder
        """
    )
    ap.add_argument(
        "--folder",
        type=str,
        default="src/models/conveyor/dataset/train/potato",
        help="Folder containing potato files to rename (default: src/models/conveyor/dataset/train/potato)"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without actually renaming files"
    )
    
    args = ap.parse_args()
    
    folder = Path(args.folder)
    rename_potato_files(folder=folder, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

