"""
Rename files in a folder by adding a prefix to filenames.

Usage (from repo root or from src/):
  py src/tools/rename_with_prefix.py --prefix fg
  py src/tools/rename_with_prefix.py --folder src/tools/potato --prefix fg
  py src/tools/rename_with_prefix.py --folder src/tools/potato --prefix fg --pattern "^\\d+-\\d+"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def rename_files_with_prefix(folder: Path, prefix: str, pattern: str | None = None, dry_run: bool = False):
    """
    Rename files in a folder by adding a prefix to filenames.
    
    Args:
        folder: Directory containing files to rename
        prefix: Prefix to add to filenames
        pattern: Optional regex pattern to match filenames (without extension). 
                 If None, renames ALL files.
        dry_run: If True, only print what would be renamed without actually renaming
    """
    if not folder.is_dir():
        raise SystemExit(f"Error: '{folder}' is not a directory.")
    
    # Get all files in the folder
    files = [f for f in folder.iterdir() if f.is_file()]
    
    if not files:
        print(f"No files found in '{folder}'")
        return
    
    # Compile pattern if provided, otherwise match all files
    pattern_re = None
    if pattern is not None:
        pattern_re = re.compile(pattern)
    
    renamed_count = 0
    skipped_count = 0
    
    print(f"Scanning folder: {folder.resolve()}")
    if pattern_re:
        print(f"Pattern: {pattern}")
    else:
        print(f"Pattern: ALL FILES (no filter)")
    print(f"Prefix: {prefix}")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'RENAME'}")
    print("-" * 60)
    
    for file_path in sorted(files):
        # Get filename without extension
        stem = file_path.stem
        suffix = file_path.suffix
        
        # Check if filename matches pattern (if pattern is provided)
        if pattern_re is not None and not pattern_re.match(stem):
            skipped_count += 1
            continue
        
        # Check if already has prefix
        if stem.startswith(prefix):
            print(f"SKIP (already prefixed): {file_path.name}")
            skipped_count += 1
            continue
        
        # Create new filename
        new_stem = prefix + stem
        new_path = file_path.parent / (new_stem + suffix)
        
        # Check if target already exists
        if new_path.exists():
            print(f"SKIP (target exists): {file_path.name} -> {new_path.name}")
            skipped_count += 1
            continue
        
        if dry_run:
            print(f"WOULD RENAME: {file_path.name} -> {new_path.name}")
        else:
            try:
                file_path.rename(new_path)
                print(f"RENAMED: {file_path.name} -> {new_path.name}")
                renamed_count += 1
            except Exception as e:
                print(f"ERROR renaming {file_path.name}: {e}")
                skipped_count += 1
    
    print("-" * 60)
    print(f"Summary: {renamed_count} renamed, {skipped_count} skipped")
    
    if dry_run and renamed_count > 0:
        print(f"\nRun without --dry-run to apply these changes.")


def main():
    ap = argparse.ArgumentParser(
        description="Rename files by adding a prefix to filenames matching a pattern",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
  # Add prefix "fg" to ALL files in default potato folder
  py src/tools/rename_with_prefix.py --prefix fg
  
  # Add prefix "fg" to files matching "01-24" pattern only
  py src/tools/rename_with_prefix.py --prefix fg --pattern "^\d+-\d+"
  
  # Dry run to preview changes
  py src/tools/rename_with_prefix.py --prefix fg --dry-run
        """
    )
    ap.add_argument(
        "--folder",
        type=str,
        default="src/tools/potato",
        help="Folder containing files to rename (default: src/tools/potato)"
    )
    ap.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Prefix to add to matching filenames (e.g., 'fg')"
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Optional regex pattern to match filenames (without extension). If not provided, renames ALL files."
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without actually renaming files"
    )
    
    args = ap.parse_args()
    
    folder = Path(args.folder)
    rename_files_with_prefix(
        folder=folder,
        prefix=args.prefix,
        pattern=args.pattern,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()

