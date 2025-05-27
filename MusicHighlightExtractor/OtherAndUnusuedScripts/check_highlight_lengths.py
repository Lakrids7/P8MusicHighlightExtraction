import os
import argparse
import traceback
import pickle
from multiprocessing import Pool, cpu_count
import time
import librosa # To get audio duration
import math # For checking float equality with tolerance

# --- Constants ---
PREVIEW_FILENAME = "original_preview.mp3"
DEFAULT_CACHE_DIR = "preview_length_cache" # Can reuse the same cache
TARGET_DURATION = 30.0 # Default target duration in seconds

# --- Helper function for multiprocessing ---
def get_preview_duration_for_song(args):
    """
    Gets the duration of the preview MP3 for a single song and returns
    it along with the song directory name. Uses caching.
    """
    song_dir_name, root_dir, cache_dir = args

    item_path = os.path.join(root_dir, song_dir_name)
    preview_audio_path = os.path.join(item_path, PREVIEW_FILENAME)
    cache_path = os.path.join(cache_dir, f'{song_dir_name}_preview_len_cache.pkl')

    # 1. Check cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                if isinstance(cached_data, (float, int, type(None))):
                    return song_dir_name, cached_data # Return name and cached duration/None
                else:
                    print(f"Warning: Invalid data format in cache {cache_path}. Recalculating.")
        except Exception as e:
            print(f"Warning: Error loading cache {cache_path}: {e}. Recalculating.")

    # 2. Process preview file
    duration = None
    try:
        if not os.path.exists(preview_audio_path):
             # File missing, duration remains None
             pass
        else:
            duration = librosa.get_duration(path=preview_audio_path)
            if duration is None or duration <= 0:
                 print(f"Warning: Could not get valid duration for {preview_audio_path} (Duration: {duration}).")
                 duration = None # Treat as invalid

    except Exception as e:
        print(f"--- Error processing preview duration for {song_dir_name} ---")
        print(f"   File: {preview_audio_path}")
        # print(f"   Error: {e}") # Keep it concise unless debugging
        duration = None # Ensure None on error

    # 3. Save result to cache
    try:
         with open(cache_path, 'wb') as f:
             pickle.dump(duration, f)
    except Exception as e:
         print(f"Warning: Could not save preview duration cache {cache_path}: {e}")

    return song_dir_name, duration # Return name and duration/None

# --- Data Collection Function (Identifies non-target durations) ---
def find_non_target_duration_previews(dataset_folder, cache_dir=DEFAULT_CACHE_DIR, target_duration=TARGET_DURATION, tolerance=0.1):
    """
    Finds songs whose preview duration is outside the target range (target +/- tolerance).
    Uses multiprocessing and caching.

    Returns:
        list: A list of tuples, where each tuple is (song_dir_name, actual_duration).
    """
    print(f"Scanning for '{PREVIEW_FILENAME}' files with duration outside {target_duration:.1f}s +/- {tolerance:.2f}s...")
    print(f"Checking durations in range ({target_duration - tolerance:.2f}s, {target_duration + tolerance:.2f}s)")
    os.makedirs(cache_dir, exist_ok=True)

    all_items = os.listdir(dataset_folder)
    song_dirs = [item for item in all_items if os.path.isdir(os.path.join(dataset_folder, item))]
    valid_song_dirs = song_dirs # Assume all dirs are potential song dirs for this check

    print(f"Found {len(valid_song_dirs)} potential song directories to check.")

    tasks = [(song_dir, dataset_folder, cache_dir) for song_dir in valid_song_dirs]

    exceptions = [] # List to store (song_dir_name, actual_duration) for non-matches
    processed_count = 0
    found_preview_count = 0
    start_time = time.time()

    num_workers = min(cpu_count(), 8)
    print(f"Using {num_workers} worker processes.")

    min_allowed = target_duration - tolerance
    max_allowed = target_duration + tolerance

    try:
        with Pool(num_workers) as pool:
            results_iterator = pool.imap_unordered(get_preview_duration_for_song, tasks)

            for i, (song_dir_name, duration) in enumerate(results_iterator):
                processed_count += 1
                if duration is not None:
                    found_preview_count += 1
                    # Check if duration is outside the allowed range
                    # Using math.isclose is safer for float comparisons, but checking bounds is clearer here
                    if not (min_allowed <= duration <= max_allowed):
                        exceptions.append((song_dir_name, duration))
                # else: Duration was None (file missing or error), ignore for this check

                if (i + 1) % 200 == 0: # Progress indicator
                    print(f"  Checked {i + 1}/{len(tasks)} directories...")

    except Exception as e:
        print(f"\nAn error occurred during parallel processing: {e}")
        traceback.print_exc()
        print("Attempting to report findings based on data collected so far...")

    end_time = time.time()
    print(f"\nFinished scan in {end_time - start_time:.2f} seconds.")
    print(f"Checked {processed_count} directories.")
    print(f"Found {found_preview_count} '{PREVIEW_FILENAME}' files with valid durations.")
    print(f"Identified {len(exceptions)} previews with duration outside the target range.")

    # Sort exceptions alphabetically by song directory name
    exceptions.sort(key=lambda item: item[0])

    return exceptions

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Find songs whose '{PREVIEW_FILENAME}' duration is not close to a target value.")
    parser.add_argument('--dataset_folder', type=str, required=True,
                        help="Path to the main dataset folder containing song subdirectories")
    parser.add_argument('--target_duration', type=float, default=TARGET_DURATION,
                        help=f"The target duration in seconds (default: {TARGET_DURATION})")
    parser.add_argument('--tolerance', type=float, default=0.1,
                        help="Allowed deviation from target duration in seconds (default: 0.1)")
    parser.add_argument('--cache_dir', type=str, default=DEFAULT_CACHE_DIR,
                        help="Directory to cache collected preview durations")
    parser.add_argument('--output_file', type=str, default=None,
                        help="Optional: Path to save the list of non-matching songs (one per line)")

    args = parser.parse_args()

    if not os.path.isdir(args.dataset_folder):
        print(f"Error: Dataset folder not found at {args.dataset_folder}")
    else:
        if args.tolerance < 0:
            print("Error: Tolerance must be non-negative.")
        else:
            # Find the exceptions
            non_target_previews = find_non_target_duration_previews(
                args.dataset_folder,
                args.cache_dir,
                args.target_duration,
                args.tolerance
            )

            # Report the results
            if not non_target_previews:
                print(f"\nAll found '{PREVIEW_FILENAME}' files have durations within {args.target_duration:.1f}s +/- {args.tolerance:.2f}s.")
            else:
                print(f"\n--- Songs with '{PREVIEW_FILENAME}' duration outside ({args.target_duration - args.tolerance:.2f}s, {args.target_duration + args.tolerance:.2f}s) ---")
                for song_dir, duration in non_target_previews:
                    print(f"- {song_dir}: {duration:.3f}s")

                # Optionally save to file
                if args.output_file:
                    try:
                        # Ensure output directory exists
                        output_dir = os.path.dirname(args.output_file)
                        if output_dir:
                            os.makedirs(output_dir, exist_ok=True)

                        with open(args.output_file, 'w', encoding='utf-8') as f:
                            f.write(f"# Previews with duration outside {args.target_duration:.1f}s +/- {args.tolerance:.2f}s\n")
                            f.write("# Format: song_directory_name: duration_in_seconds\n")
                            for song_dir, duration in non_target_previews:
                                f.write(f"{song_dir}: {duration:.3f}\n")
                        print(f"\nList saved to: {args.output_file}")
                    except Exception as e:
                        print(f"\nError saving list to file '{args.output_file}': {e}")