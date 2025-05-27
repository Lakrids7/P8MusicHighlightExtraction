#!/usr/bin/env python3
# make_split.py
import json
import random
import pathlib
import re
import argparse

def collect_song_dirs(root, max_dtw=1200.0):
    """
    Scan `root/*/metadata.txt` and return a list of sub‑folder paths
    whose DTW Cost is <= max_dtw.
    """
    dirs = []
    for song_dir in pathlib.Path(root).iterdir():
        if not song_dir.is_dir():
            continue
        meta = song_dir / "metadata.txt"
        if not meta.exists():
            continue

        try:
            with meta.open() as fh:
                for line in fh:
                    if line.lower().startswith("dtw cost"):
                        cost = float(line.split(":", 1)[1].strip())
                        if cost <= max_dtw:
                            # Use relative paths if root is provided, else absolute
                            # Convert to string for JSON serialization
                            dirs.append(str(song_dir.resolve())) 
                        break
        except Exception as e:
            print(f"[warn] skipping {song_dir}: {e}")
    return dirs

def train_val_test_split(items, seed=42):
    """
    Splits items into train, validation, and test sets (70/15/15).

    Args:
        items (list): The list of items to split.
        seed (int): The random seed for shuffling.

    Returns:
        tuple: A tuple containing (train_items, val_items, test_items).
    """
    if not items:
        return [], [], []

    # Shuffle the items deterministically
    items_copy = list(items) # Make a copy to avoid modifying the original list
    random.Random(seed).shuffle(items_copy)

    n = len(items_copy)
    # Calculate split points
    # Train ends at 70%
    cut1 = int(n * 0.70)
    # Validation ends at 70% + 15% = 85%
    cut2 = int(n * 0.85) # Make sure this is relative to n, not cut1

    train_items = items_copy[:cut1]
    val_items = items_copy[cut1:cut2]
    test_items = items_copy[cut2:]

    return train_items, val_items, test_items

def main():
    parser = argparse.ArgumentParser(
        description="Create train/val/test split (70/15/15) JSON for songs with DTW Cost <= threshold.")
    parser.add_argument("--root", default="NewDataUnfiltered",
                        help="root directory that contains song sub‑folders")
    parser.add_argument("--outfile", default="train_val_test_split.json",
                        help="where to save the resulting JSON (default: train_val_test_split.json)")
    parser.add_argument("--max_dtw", type=float, default=1200.0,
                        help="maximum DTW Cost to keep a song (default: 1200.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="PRNG seed for deterministic shuffling (default: 42)")
    args = parser.parse_args()

    print(f"Scanning '{args.root}' for songs with DTW Cost <= {args.max_dtw}...")
    songs = collect_song_dirs(args.root, args.max_dtw)
    
    if not songs:
        print(f"[error] No songs matched the DTW Cost criterion (<= {args.max_dtw}) in '{args.root}'. Exiting.")
        exit(1) # Use exit(1) to indicate an error

    print(f"Found {len(songs)} eligible songs. Splitting into Train (70%), Validation (15%), Test (15%)...")
    train, val, test = train_val_test_split(songs, args.seed)

    # Ensure the split covers all songs and proportions are roughly correct
    assert len(train) + len(val) + len(test) == len(songs), "Split sizes don't sum to total!"

    split = {"train": train, "val": val, "test": test}
    
    try:
        output_path = pathlib.Path(args.outfile)
        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True) 
        
        with output_path.open("w") as fp:
            json.dump(split, fp, indent=2)
            
        print(f"\nSaved split to '{args.outfile}':")
        print(f"  Train: {len(train)} items ({len(train)/len(songs)*100:.1f}%)")
        print(f"  Validation: {len(val)} items ({len(val)/len(songs)*100:.1f}%)")
        print(f"  Test: {len(test)} items ({len(test)/len(songs)*100:.1f}%)")
        print(f"  Total: {len(songs)} items")

    except IOError as e:
        print(f"[error] Could not write to output file '{args.outfile}': {e}")
        exit(1)
    except Exception as e:
        print(f"[error] An unexpected error occurred: {e}")
        exit(1)

if __name__ == "__main__":
    main()