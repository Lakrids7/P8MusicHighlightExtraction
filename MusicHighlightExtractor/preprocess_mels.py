# preprocess_mels.py
import os
import argparse
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm # For progress bar
import re # For extracting Salami ID from filename

# --- Constants ---
SR, HOP, WIN, N_MELS = 22_050, 512, 2_048, 128
CHUNK_FRAMES = 128 # As defined in your training script
OUTPUT_MEL_FILENAME = "full_song.npy" # Expected by the training script
EXPECTED_AUDIO_EXTENSION_IN_DIR = ".wav" # The format of files in salami_songs_base_dir

# --- Function to process audio ---
def audio_to_full_chunks_and_save(audio_path: str, output_npy_path: str):
    """Loads audio, computes mel spec chunks, saves as .npy."""
    try:
        y, sr_loaded = librosa.load(audio_path, sr=SR, mono=True)
        if sr_loaded != SR:
            # This case should ideally not happen if sr=SR is passed to load,
            # but good to be aware of. Librosa resamples by default.
            print(f"Warning: Audio {os.path.basename(audio_path)} loaded with sr {sr_loaded} instead of {SR}. Resampling occurred.")

        # Pad if too short for even one full chunk
        min_samples_for_one_chunk_mel = CHUNK_FRAMES * HOP + WIN # Samples needed for librosa.feature.melspectrogram input
        
        if len(y) < min_samples_for_one_chunk_mel:
            # print(f"Info: Padding {os.path.basename(audio_path)} (len {len(y)}) to min_samples {min_samples_for_one_chunk_mel}")
            y_padded = librosa.util.pad_center(y, size=min_samples_for_one_chunk_mel)
            y = y_padded

        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=WIN, hop_length=HOP, n_mels=N_MELS)
        mel = librosa.power_to_db(mel, ref=np.max).T # Transpose to (Time, Mels)

        n_frames_total = mel.shape[0]
        
        if n_frames_total < CHUNK_FRAMES:
            # This might happen if padding was just enough for melspectrogram, but resulting frames are still < CHUNK_FRAMES
            # print(f"Warning: {os.path.basename(audio_path)} too short after Mel ({n_frames_total} frames vs {CHUNK_FRAMES} needed). Skipping.")
            return False

        n_chunks = n_frames_total // CHUNK_FRAMES
        if n_chunks == 0: # Should be caught by above, but defensive
            # print(f"Warning: {os.path.basename(audio_path)} resulted in 0 chunks. Skipping.")
            return False

        # Trim to an exact multiple of CHUNK_FRAMES
        mel = mel[:n_chunks * CHUNK_FRAMES]
        # Reshape into (n_chunks, CHUNK_FRAMES, N_MELS)
        mel = mel.reshape(n_chunks, CHUNK_FRAMES, N_MELS)
        # Transpose to (n_chunks, N_MELS, CHUNK_FRAMES) as expected by some models
        mel_chunks = mel.transpose(0, 2, 1).astype(np.float32)

        os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
        np.save(output_npy_path, mel_chunks)
        return True

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        # import traceback
        # traceback.print_exc()
        return False

def extract_salami_id_from_filename_for_debug(filename: str) -> str | None:
    """
    Extracts SALAMI ID (numeric part) from filenames like 'salami_ID_details.ANY_EXT'.
    Relies on the 'salami_ID_' prefix structure.
    """
    # Example: salami_95_K8TEWwu9-98.wav -> 95
    # Example: salami_954_FTRX-_gv8tM.mp3 -> 954
    match = re.match(r"salami_(\d+)_.*", filename, re.IGNORECASE)
    if match:
        return match.group(1)
    # print(f"  Debug - extract_salami_id_for_debug: No match for '{filename}' with r'salami_(\\d+)_.*'")
    return None

def debug_csv_and_song_files_mismatch(csv_path: str, salami_songs_base_dir: str):
    """
    Compares song IDs found in the CSV with song IDs derived from filenames
    in the salami_songs_base_dir to identify discrepancies.
    """
    print("\n--- Debugging Song and CSV Mismatch ---")

    # 1. Get Salami IDs from the CSV
    try:
        df_csv = pd.read_csv(csv_path)
        if 'salami_id' not in df_csv.columns:
            print(f"Error: 'salami_id' column not found in {csv_path}. Cannot perform debug comparison.")
            return
        csv_salami_ids = set(df_csv['salami_id'].astype(str).unique())
        print(f"Found {len(csv_salami_ids)} unique Salami IDs in CSV: {csv_path}")
    except Exception as e:
        print(f"Error reading or processing CSV {csv_path} for debugging: {e}")
        return

    # 2. Get Salami IDs from song filenames in the directory
    if not os.path.isdir(salami_songs_base_dir):
        print(f"Error: Songs directory not found: {salami_songs_base_dir}. Cannot perform debug comparison.")
        return

    dir_salami_ids = set()
    found_files_count = 0
    parsed_ids_from_dir_count = 0
    unparsable_filenames_in_dir = []

    for filename_in_dir in os.listdir(salami_songs_base_dir):
        # Check if the file has the expected extension and starts with "salami_"
        if filename_in_dir.lower().endswith(EXPECTED_AUDIO_EXTENSION_IN_DIR) and \
           filename_in_dir.lower().startswith("salami_"):
            found_files_count += 1
            s_id = extract_salami_id_from_filename_for_debug(filename_in_dir)
            if s_id:
                dir_salami_ids.add(s_id)
                parsed_ids_from_dir_count += 1
            else:
                unparsable_filenames_in_dir.append(filename_in_dir)
    
    print(f"Found {found_files_count} '{EXPECTED_AUDIO_EXTENSION_IN_DIR}' files starting with 'salami_' in directory: {salami_songs_base_dir}")
    print(f"Derived {len(dir_salami_ids)} unique Salami IDs from these filenames (parsed {parsed_ids_from_dir_count} successfully).")
    if unparsable_filenames_in_dir:
        print(f"  Warning (debug): Could not parse Salami ID from {len(unparsable_filenames_in_dir)} filenames in directory. Examples: {unparsable_filenames_in_dir[:5]}")


    # 3. Compare the two sets
    ids_in_dir_not_in_csv = dir_salami_ids - csv_salami_ids
    ids_in_csv_not_in_dir = csv_salami_ids - dir_salami_ids

    if ids_in_dir_not_in_csv:
        print(f"\nINFO: {len(ids_in_dir_not_in_csv)} Salami IDs found in directory '{salami_songs_base_dir}' but NOT in CSV '{csv_path}'. (Sample: {sorted(list(ids_in_dir_not_in_csv), key=lambda x: int(x) if x.isdigit() else x)[:5]})")
        print("      These songs will NOT be processed by the current script because they are not listed in the CSV.")
    else:
        print(f"\nGood: All Salami IDs derived from song files in '{salami_songs_base_dir}' appear to be present in the CSV.")

    if ids_in_csv_not_in_dir:
        print(f"\nWARNING: {len(ids_in_csv_not_in_dir)} Salami IDs found in CSV '{csv_path}' but no corresponding parsable file found in directory '{salami_songs_base_dir}'. (Sample: {sorted(list(ids_in_csv_not_in_dir), key=lambda x: int(x) if x.isdigit() else x)[:5]})")
        print("         These CSV entries will likely cause 'Audio file not found' errors during processing.")
    else:
        print(f"\nGood: All Salami IDs in the CSV appear to have corresponding parsable filenames in '{salami_songs_base_dir}'.")
    
    print("--- End of Debugging ---")


def main():
    p = argparse.ArgumentParser(description="Pre-compute Mel spectrogram chunks for SALAMI chorus data")
    p.add_argument("--csv_path", default="salami_chorus_annotations_with_duration.csv",
                   help="Path to the CSV file with song paths and salami_ids.")
    p.add_argument("--salami_songs_base_dir", default="salami-data-public-master/transformed_audio/", # Default to where .wavs are
                   help="Base directory where SALAMI audio files (expected format: .wav) are located.")
    p.add_argument("--output_mel_base_dir", default="data/salami_mels",
                   help="Base directory to save precomputed Mel spectrograms.")
    p.add_argument("--force_recompute", action="store_true",
                   help="Force recomputation even if output file exists.")
    args = p.parse_args()

    # --- Call the debugging function ---
    debug_csv_and_song_files_mismatch(args.csv_path, args.salami_songs_base_dir)
    # --- End of debugging call ---

    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found at {args.csv_path}")
        return

    try:
        df = pd.read_csv(args.csv_path)
    except Exception as e:
        print(f"Error reading CSV {args.csv_path}: {e}")
        return

    # Ensure 'salami_id' is present, as the script relies on it.
    if 'salami_id' not in df.columns:
        print(f"Error: 'salami_id' column is missing from the CSV file: {args.csv_path}")
        return
        
    # Ensure 'song_filename' or 'song_filepath_abs' are present for path construction
    if not ('song_filename' in df.columns or 'song_filepath_abs' in df.columns):
        print(f"Error: CSV must contain either 'song_filename' or 'song_filepath_abs' column. Found: {df.columns.tolist()}")
        return

    unique_songs_df = df.drop_duplicates(subset=['salami_id']).copy()
    print(f"\nFound {len(unique_songs_df)} unique songs to process *based on the CSV entries*.") 

    processed_count, error_count, skipped_count, already_exists_count = 0, 0, 0, 0

    for index, row in tqdm(unique_songs_df.iterrows(), total=unique_songs_df.shape[0], desc="Processing songs based on CSV entries"):
        salami_id = str(row['salami_id'])
        
        # --- ADAPTED PATH RESOLUTION ---
        # We need to derive the base filename (e.g., "salami_10_s08jD3E6Mpg") from the CSV,
        # then append the EXPECTED_AUDIO_EXTENSION_IN_DIR (e.g., ".wav").

        base_filename_from_csv_no_ext = None

        if 'song_filename' in row and pd.notna(row['song_filename']):
            # Prefer 'song_filename' if available
            base_filename_from_csv_no_ext = os.path.splitext(row['song_filename'])[0]
        elif 'song_filepath_abs' in row and pd.notna(row['song_filepath_abs']):
            # Fallback to 'song_filepath_abs' if 'song_filename' is missing
            base_filename_from_csv_no_ext = os.path.splitext(os.path.basename(row['song_filepath_abs']))[0]
        
        if not base_filename_from_csv_no_ext:
            print(f"Warning: Could not determine base filename for Salami ID {salami_id} from CSV row. Skipping.")
            skipped_count += 1
            continue
            
        # Construct the target filename with the expected extension for files in the directory
        target_audio_filename_in_dir = base_filename_from_csv_no_ext + EXPECTED_AUDIO_EXTENSION_IN_DIR
        audio_path = os.path.join(args.salami_songs_base_dir, target_audio_filename_in_dir)
        # --- END ADAPTED PATH RESOLUTION ---

        if not os.path.exists(audio_path):
            # Provide more context for missing files
            original_csv_filename_info = row.get('song_filename', 'N/A (song_filename missing)')
            original_csv_filepath_abs_info = row.get('song_filepath_abs', 'N/A (song_filepath_abs missing)')
            print(f"Warning: Audio file not found at constructed path: {audio_path}")
            print(f"         (For Salami ID: {salami_id})")
            print(f"         (Derived from CSV song_filename: '{original_csv_filename_info}', song_filepath_abs: '{original_csv_filepath_abs_info}')")
            skipped_count += 1
            continue

        output_song_dir = os.path.join(args.output_mel_base_dir, salami_id)
        output_npy_path = os.path.join(output_song_dir, OUTPUT_MEL_FILENAME)

        if not args.force_recompute and os.path.exists(output_npy_path):
            already_exists_count += 1
            continue

        if audio_to_full_chunks_and_save(audio_path, output_npy_path):
            processed_count += 1
        else:
            error_count += 1 # audio_to_full_chunks_and_save prints its own error/warning

    print(f"\nPreprocessing Complete (based on CSV entries).")
    print(f"  Successfully processed and saved Mel spectrograms: {processed_count}")
    print(f"  Output files already existed (skipped recomputation): {already_exists_count}")
    print(f"  Errors during Mel processing (audio load/Mel calc failed): {error_count}")
    print(f"  Skipped (e.g., audio file not found from CSV info, or audio too short): {skipped_count}")
    print(f"Precomputed Mels saved in: {args.output_mel_base_dir}")

if __name__ == "__main__":
    main()