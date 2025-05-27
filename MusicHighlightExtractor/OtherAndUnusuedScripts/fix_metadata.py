import os
import re
import argparse
import librosa
import math
import sys
import numpy as np # Import numpy

# --- Copy required functions from the original song_processing.py ---
def compute_mfcc(audio_path, sr=22050):
    """Computes MFCC features for an audio file. (Copied for recalculation)"""
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found for MFCC: {audio_path}")
        audio, sr_loaded = librosa.load(audio_path, sr=sr, mono=True)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr_loaded, n_mfcc=20, hop_length=512)
        return mfcc
    except Exception as e:
        print(f"  ❌ Error computing MFCC for {os.path.basename(audio_path)}: {e}")
        raise

def sliding_window_dtw(full_song_mfcc, snippet_mfcc, hop_length=512, sr=22050):
    """
    Performs sliding window DTW using the original manual iteration method.
    (Copied for recalculation)
    """
    snippet_len = snippet_mfcc.shape[1]
    full_len = full_song_mfcc.shape[1]

    if snippet_len == 0 or full_len == 0:
         raise ValueError("MFCC features are empty.")
    if snippet_len > full_len:
        # This could happen if original_preview is longer than full_song somehow?
        print(f"  ⚠️ Warning: Snippet length ({snippet_len}) > Full song length ({full_len}). Skipping DTW.")
        return 0, None # Cannot calculate
        # raise ValueError("Snippet MFCCs are longer than full song MFCCs.")

    step_size = 20
    best_cost = np.inf
    best_idx = 0

    snippet_norm = (snippet_mfcc - np.mean(snippet_mfcc)) / (np.std(snippet_mfcc) + 1e-8)

    print(f"    -> Recalculating DTW (Snippet len: {snippet_len}, Full len: {full_len}, Step: {step_size})")
    num_steps = (full_len - snippet_len) // step_size
    # print(f"    -> Estimated steps: {num_steps}") # Less verbose for script

    for i, idx in enumerate(range(0, full_len - snippet_len, step_size)):
        window = full_song_mfcc[:, idx:idx + snippet_len]
        window_norm = (window - np.mean(window)) / (np.std(window) + 1e-8)
        try:
            D, _ = librosa.sequence.dtw(X=snippet_norm, Y=window_norm, metric='euclidean')
            cost = D[-1, -1]
            if cost < best_cost:
                best_cost = cost
                best_idx = idx
        except Exception as dtw_err:
             # Log less verbosely in script
             if i % 100 == 0: # Log only occasionally if errors occur
                print(f"    ... Error at window {idx}: {dtw_err}")
             continue

    if best_cost == np.inf:
        print("    ⚠️ DTW recalculation did not find any valid match.")
        return 0, None

    timestamp_seconds = librosa.frames_to_time(best_idx, sr=sr, hop_length=hop_length)
    # print(f"    -> Recalculated best match at {timestamp_seconds:.2f}s with cost {best_cost:.2f}")
    return timestamp_seconds, best_cost
# --- End of copied functions ---

def parse_metadata(filepath):
    """Parses existing metadata.txt into a dictionary."""
    metadata = {}
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                match = re.match(r'([^:]+):\s*(.*)', line)
                if match:
                    metadata[match.group(1).strip()] = match.group(2).strip()
    except Exception as e:
        print(f"  ❌ Error reading metadata file {filepath}: {e}")
        return None
    return metadata

def get_audio_duration(filepath):
    """Safely gets audio duration using librosa."""
    if not os.path.exists(filepath):
        print(f"  ⚠️ Audio file not found: {filepath}")
        return None
    try:
        duration = librosa.get_duration(filename=filepath)
        return duration
    except Exception as e:
        print(f"  ❌ Error getting duration for {os.path.basename(filepath)}: {e}")
        return None

def format_duration(duration_sec):
    """Formats duration in seconds to 'XXX.XX seconds'."""
    if duration_sec is None: return "Unknown"
    try: return f"{duration_sec:.2f} seconds"
    except Exception: return "Unknown"

def process_folder(folder_path):
    """Processes a single song folder to update metadata and recalculate DTW."""
    print(f"\nProcessing folder: {os.path.basename(folder_path)}")
    metadata_file = os.path.join(folder_path, "metadata.txt")
    preview_file = os.path.join(folder_path, "original_preview.mp3") # Used for DTW recalc
    full_song_file = os.path.join(folder_path, "full_song.mp3") # Used for DTW recalc & duration

    # --- Check required files for recalculations ---
    if not os.path.exists(metadata_file):
        print("  ⚠️ metadata.txt not found. Skipping.")
        return False
    if not os.path.exists(preview_file):
        print("  ⚠️ original_preview.mp3 not found. Cannot recalculate DTW or preview duration. Skipping.")
        return False
    if not os.path.exists(full_song_file):
        print("  ⚠️ full_song.mp3 not found. Cannot recalculate DTW or full duration. Skipping.")
        return False

    # --- Parse Existing Data (for fallback and other fields) ---
    print("  🔍 Reading existing metadata...")
    old_metadata = parse_metadata(metadata_file)
    if old_metadata is None:
        print("  ❌ Failed to parse existing metadata. Skipping.")
        return False

    # Extract fields needed for the new file
    song_name = old_metadata.get("Song Name", "Unknown Song")
    artist = old_metadata.get("Artist", "Unknown Artist")
    genres = old_metadata.get("Genres", "Unknown")
    # Get original detected timestamp (we don't recalculate this, just use it for end time)
    detected_ts_str = old_metadata.get("Detected Timestamp", "0 seconds")
    try:
        ts_match = re.search(r'\d+', detected_ts_str)
        detected_start_sec = int(ts_match.group(0)) if ts_match else 0
    except (ValueError, TypeError):
        print(f"  ⚠️ Could not parse 'Detected Timestamp': '{detected_ts_str}'. Using 0.")
        detected_start_sec = 0

    print(f"    - Song: {song_name} by {artist}")
    print(f"    - Original Start Timestamp: {detected_start_sec}s")

    # --- Recalculate Values ---
    print("  ⏱️ Calculating durations...")
    preview_duration = get_audio_duration(preview_file)
    full_duration = get_audio_duration(full_song_file)

    if preview_duration is None:
        print("  ❌ Failed to get preview duration. Cannot calculate end time. Aborting update.")
        return False

    # Calculate end time based on original start + recalculated preview duration
    preview_end_time = detected_start_sec + preview_duration
    preview_end_time_rounded = math.ceil(preview_end_time)

    print(f"    - Calculated Preview Duration: {preview_duration:.2f}s")
    print(f"    - Calculated Preview End Time: {preview_end_time_rounded}s")
    print(f"    - Calculated Full Song Duration: {full_duration:.2f}s" if full_duration is not None else "Unknown")

    # --- Recalculate DTW Cost ---
    print("  🔄 Recalculating DTW cost...")
    recalculated_dtw_cost = None
    try:
        full_mfcc = compute_mfcc(full_song_file, sr=22050)
        snippet_mfcc = compute_mfcc(preview_file, sr=22050)
        # Use the copied sliding_window_dtw function
        # It returns (timestamp, cost), we only need the cost here
        _, recalculated_dtw_cost = sliding_window_dtw(full_mfcc, snippet_mfcc, sr=22050, hop_length=512)

        if recalculated_dtw_cost is not None:
            print(f"    - Recalculated DTW Cost: {recalculated_dtw_cost:.2f}")
        else:
            print("    - DTW recalculation failed or returned no cost.")

    except Exception as e:
        print(f"  ❌ Error during MFCC/DTW recalculation: {e}")
        # Keep recalculated_dtw_cost as None

    # --- Format Corrected Values ---
    print("  ✍️ Formatting metadata...")
    formatted_preview_duration = format_duration(preview_duration)
    formatted_full_duration = format_duration(full_duration)

    # Format the *recalculated* DTW cost
    if recalculated_dtw_cost is not None:
        formatted_dtw = f"{recalculated_dtw_cost:.2f}"
    else:
        formatted_dtw = "Calculation Failed" # Or "Not available"

    print(f"    - Final Formatted DTW Cost: {formatted_dtw}")

    # --- Construct New Metadata Content ---
    new_metadata_content = [
        f"Song Name: {song_name}",
        f"Artist: {artist}",
        f"Detected Timestamp: {detected_start_sec} seconds", # Keep original start
        f"Preview End Timestamp: {preview_end_time_rounded} seconds", # New end time
        f"Preview Duration: {formatted_preview_duration}", # New preview duration
        f"DTW Cost: {formatted_dtw}", # Use recalculated & formatted value
        f"Full Song Duration: {formatted_full_duration}", # New full duration
        f"Genres: {genres}"
    ]
    # Optional: Add back Spotify URL if it existed
    if "Spotify Track URL" in old_metadata:
         new_metadata_content.append(f"Spotify Track URL: {old_metadata['Spotify Track URL']}")

    # --- Write Updated Metadata ---
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_metadata_content) + '\n')
        print(f"  ✅ Successfully updated {metadata_file}")
        return True
    except Exception as e:
        print(f"  ❌ Error writing updated metadata to {metadata_file}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Update metadata.txt files in song folders, recalculating DTW cost and durations.")
    parser.add_argument("data_folder", help="Path to the main directory containing song subfolders (e.g., Song_data_New)")
    args = parser.parse_args()

    parent_folder = args.data_folder
    if not os.path.isdir(parent_folder):
        print(f"Error: Folder not found: {parent_folder}")
        sys.exit(1)

    print(f"Starting metadata update process in: {parent_folder}")
    updated_count = 0
    skipped_count = 0

    for item_name in sorted(os.listdir(parent_folder)): # Sort for predictable order
        item_path = os.path.join(parent_folder, item_name)
        if os.path.isdir(item_path):
            if process_folder(item_path):
                updated_count += 1
            else:
                skipped_count += 1
        # else: # Ignore non-directories silently now
        #     print(f"\nSkipping non-directory item: {item_name}")

    print("\n--------------------")
    print("Metadata update complete.")
    print(f"Folders processed successfully: {updated_count}")
    print(f"Folders skipped due to errors or missing files: {skipped_count}")
    print("--------------------")

if __name__ == "__main__":
    # Ensure necessary libraries are installed: pip install librosa numpy
    main()