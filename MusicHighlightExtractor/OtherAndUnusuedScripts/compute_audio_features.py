import argparse
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re # For parsing the metadata file

# --- Constants for Feature Extraction ---
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
WINDOW_TYPE = 'hamming'
ROLL_PERCENT = 0.85
DTW_COST_THRESHOLD = 1200.0
MAX_SONGS_TO_PROCESS = 100 # <<< Maximum number of songs to process

def parse_metadata(metadata_file_path):
    """Parses the metadata.txt file to extract DTW Cost and Preview Timestamp."""
    if not metadata_file_path.is_file():
        return None, None

    dtw_cost = None
    preview_timestamp = None
    try:
        with open(metadata_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("DTW Cost:"):
                    match = re.search(r"DTW Cost:\s*([\d.]+)", line)
                    if match:
                        dtw_cost = float(match.group(1))
                elif line.startswith("Detected Timestamp:"):
                    match = re.search(r"Detected Timestamp:\s*([\d.]+)\s*seconds", line)
                    if match:
                        preview_timestamp = float(match.group(1))
                if dtw_cost is not None and preview_timestamp is not None:
                    break
    except Exception as e:
        print(f"    Error reading metadata file {metadata_file_path.name}: {e}")
        return None, None
    return dtw_cost, preview_timestamp

def compute_features(y, sr):
    """Computes spectral centroid, roll-off, and RMS energy."""
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, window=WINDOW_TYPE)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, window=WINDOW_TYPE, roll_percent=ROLL_PERCENT)
    rms_energy = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    return cent.squeeze(), rolloff.squeeze(), rms_energy.squeeze()

def find_max_feature_timestamps(features, times):
    """Finds the timestamp of the maximum value for each feature."""
    cent, rolloff, rms_energy = features
    max_times = {}
    if len(cent) > 0: max_times['centroid'] = times[np.argmax(cent)]
    else: max_times['centroid'] = None
    if len(rolloff) > 0: max_times['rolloff'] = times[np.argmax(rolloff)]
    else: max_times['rolloff'] = None
    if len(rms_energy) > 0: max_times['rms_energy'] = times[np.argmax(rms_energy)]
    else: max_times['rms_energy'] = None
    return max_times

def plot_features(features, times, out_path_base, song_folder_name, max_feature_times=None, ground_truth_highlight_time=None):
    """Plots the extracted features over time, optionally marking max feature times and ground truth."""
    cent, rolloff, rms_energy = features
    n_features = 3
    fig, axs = plt.subplots(n_features, 1, figsize=(12, 2 * n_features + 2), sharex=True)
    fig.suptitle(f'Audio Features for Song in Folder: {song_folder_name}', fontsize=16)

    feature_names = ['centroid', 'rolloff', 'rms_energy']
    plot_labels = ['Spectral Centroid', f'Spectral Roll-off ({ROLL_PERCENT*100:.0f}%)', 'RMS Energy']
    plot_colors = ['b', 'g', 'r']
    y_labels = ['Frequency (Hz)', 'Frequency (Hz)', 'Amplitude']
    titles = ['Spectral Centroid (Brightness)', 'Spectral Roll-off', 'RMS Energy (Loudness)']

    for i in range(n_features):
        feature_data = features[i]
        axs[i].plot(times, feature_data, label=plot_labels[i], color=plot_colors[i])
        axs[i].set_ylabel(y_labels[i]); axs[i].set_title(titles[i])
        axs[i].grid(True, linestyle=':', alpha=0.7)
        if max_feature_times and max_feature_times.get(feature_names[i]) is not None:
            max_t = max_feature_times[feature_names[i]]
            axs[i].axvline(max_t, color='black', linestyle=':', linewidth=1.5, label=f'Max {plot_labels[i].split(" ")[0]}')
        if ground_truth_highlight_time is not None:
            axs[i].axvline(ground_truth_highlight_time, color='magenta', linestyle='--', linewidth=2, label='Ground Truth Highlight')
        if feature_names[i] == 'rms_energy':
            axs2_twin = axs[i].twinx()
            rms_energy_db = librosa.amplitude_to_db(rms_energy, ref=np.max)
            axs2_twin.plot(times, rms_energy_db, label='RMS Energy (dB)', color='darkred', linestyle=':', alpha=0.6)
            axs2_twin.set_ylabel('dB')
            lines, labels_ax = axs[i].get_legend_handles_labels()
            lines2, labels_twin = axs2_twin.get_legend_handles_labels()
            axs[i].legend(lines + lines2, labels_ax + labels_twin, loc='upper right', fontsize='small')
        else:
            axs[i].legend(loc='upper right', fontsize='small')

    axs[n_features-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = out_path_base / f"{song_folder_name}_audio_features_highlighted.png"
    fig.savefig(plot_filename, dpi=150); plt.close(fig)
    print(f"    Saved highlighted feature plot to: {plot_filename.name}")

def main():
    parser = argparse.ArgumentParser(description="Compute and visualize audio features for songs, filtered by DTW cost, up to a max number.")
    parser.add_argument("in_dir", help="Parent directory containing song subfolders.")
    parser.add_argument("out_dir", help="Output directory to save feature plots.")
    parser.add_argument("--max_songs", type=int, default=MAX_SONGS_TO_PROCESS,
                        help=f"Maximum number of songs to process (default: {MAX_SONGS_TO_PROCESS}). Set to 0 for no limit.")
    args = parser.parse_args()

    in_p = Path(args.in_dir)
    out_p = Path(args.out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    if not in_p.is_dir():
        print(f"Error: Input directory '{in_p}' not found.")
        return

    current_max_songs = args.max_songs if args.max_songs > 0 else float('inf') # Use infinity if 0 for no limit

    print(f"Processing song subfolders from: {in_p}")
    print(f"Saving plots to: {out_p}")
    print(f"Using parameters: SR={SR}, N_FFT={N_FFT}, HOP_LENGTH={HOP_LENGTH}, Window={WINDOW_TYPE}, Roll-off={ROLL_PERCENT*100}%")
    print(f"DTW Cost Threshold for processing: <= {DTW_COST_THRESHOLD}")
    if current_max_songs != float('inf'):
        print(f"Processing a maximum of {current_max_songs} songs.")
    else:
        print("Processing all found songs (no maximum limit).")


    song_folders = [f for f in in_p.iterdir() if f.is_dir()]
    # Optional: Sort song_folders if you want a consistent "first N"
    # song_folders.sort()

    if not song_folders:
        print(f"No song subfolders found in '{in_p}'.")
        return

    processed_count = 0
    actually_processed_this_run = 0 # Tracks songs fully processed in this specific run
    skipped_dtw_count = 0
    skipped_other_count = 0
    comparison_results = []

    # Iterate through all found song folders initially to count
    total_song_folders_found = len(song_folders)
    print(f"Found {total_song_folders_found} song folders in total.")


    for i, song_folder in enumerate(song_folders):
        if actually_processed_this_run >= current_max_songs and current_max_songs != float('inf'): # Check if limit is reached
            print(f"\nReached maximum limit of {current_max_songs} songs to process for this run.")
            break

        print(f"\nChecking song folder {i+1}/{total_song_folders_found}: {song_folder.name}")

        audio_file = song_folder / "full_song.mp3"
        metadata_file = song_folder / "metadata.txt"

        if not audio_file.is_file():
            print(f"  Skipping {song_folder.name}: 'full_song.mp3' not found.")
            skipped_other_count += 1
            continue
        if not metadata_file.is_file():
            print(f"  Skipping {song_folder.name}: 'metadata.txt' not found.")
            skipped_other_count += 1
            continue

        dtw_cost, ground_truth_highlight_time = parse_metadata(metadata_file)

        if dtw_cost is None:
            print(f"  Skipping {song_folder.name}: Could not read DTW Cost from {metadata_file.name}.")
            skipped_other_count += 1
            continue
        if ground_truth_highlight_time is None:
            print(f"  Skipping {song_folder.name}: Could not read 'Detected Timestamp' from {metadata_file.name}.")
            skipped_other_count += 1
            continue

        if dtw_cost > DTW_COST_THRESHOLD:
            print(f"  Skipping {song_folder.name}: DTW Cost ({dtw_cost:.2f}) is above threshold ({DTW_COST_THRESHOLD}).")
            skipped_dtw_count += 1
            continue

        # If we reach here, the song passes the DTW filter and files exist.
        # This song will be counted towards the MAX_SONGS_TO_PROCESS if it gets fully processed.

        print(f"  Processing song in {song_folder.name} (DTW Cost: {dtw_cost:.2f}, Ground Truth Highlight: {ground_truth_highlight_time:.2f}s)")
        processed_count += 1 # Counts songs that passed initial checks and DTW filter
        try:
            y, sr_loaded = librosa.load(audio_file, sr=SR, mono=True)
            if y is None or len(y) == 0:
                print("    Skipped (failed to load or empty audio).")
                skipped_other_count +=1
                processed_count -=1 # Decrement as it wasn't fully processed
                continue

            print("    Computing features...")
            features = compute_features(y, sr_loaded)
            cent, rolloff, rms_energy = features
            frames = np.arange(len(cent))
            times = librosa.frames_to_time(frames, sr=sr_loaded, hop_length=HOP_LENGTH, n_fft=N_FFT)

            if len(times) == 0:
                print("    Skipped (no time frames generated for features).")
                skipped_other_count +=1
                processed_count -=1 # Decrement
                continue

            print("    Finding max feature timestamps...")
            max_feature_times = find_max_feature_timestamps(features, times)
            song_result = {
                "song_folder": song_folder.name,
                "ground_truth_highlight_s": ground_truth_highlight_time,
                "max_centroid_s": max_feature_times.get('centroid'),
                "max_rolloff_s": max_feature_times.get('rolloff'),
                "max_rms_energy_s": max_feature_times.get('rms_energy'),
                "dtw_cost": dtw_cost
            }
            comparison_results.append(song_result)
            print(f"      Max Centroid Time: {max_feature_times.get('centroid'):.2f}s" if max_feature_times.get('centroid') is not None else "      Max Centroid Time: N/A")
            print(f"      Max Roll-off Time: {max_feature_times.get('rolloff'):.2f}s" if max_feature_times.get('rolloff') is not None else "      Max Roll-off Time: N/A")
            print(f"      Max RMS Energy Time: {max_feature_times.get('rms_energy'):.2f}s" if max_feature_times.get('rms_energy') is not None else "      Max RMS Energy Time: N/A")

            print("    Plotting features...")
            plot_features(features, times, out_p, song_folder.name, max_feature_times, ground_truth_highlight_time)

            actually_processed_this_run += 1 # Increment only after successful full processing of a song

        except Exception as e:
            print(f"    Error processing song in {song_folder.name}: {e}")
            skipped_other_count +=1
            processed_count -=1 # Decrement as it wasn't fully processed due to error

    print(f"\nFeature processing complete for this run.")
    print(f"Successfully processed and plotted features for {actually_processed_this_run} songs.")
    print(f"Total songs that passed initial checks and DTW filter (attempted processing): {processed_count}")
    print(f"Skipped {skipped_dtw_count} songs due to DTW cost above threshold.")
    if skipped_other_count > 0:
        print(f"Skipped {skipped_other_count} songs due to other reasons (missing files, load/processing error).")

    if comparison_results:
        print("\n--- Comparison Summary (for successfully processed songs) ---")
        print(f"{'Song Folder':<40} {'Ground Truth (s)':<18} {'Max Centroid (s)':<18} {'Max Rolloff (s)':<18} {'Max RMS (s)':<15} {'DTW Cost':<10}")
        print("-" * 120)
        for res in comparison_results:
            gt = f"{res['ground_truth_highlight_s']:.2f}" if res['ground_truth_highlight_s'] is not None else "N/A"
            mc = f"{res['max_centroid_s']:.2f}" if res['max_centroid_s'] is not None else "N/A"
            mr = f"{res['max_rolloff_s']:.2f}" if res['max_rolloff_s'] is not None else "N/A"
            mrms = f"{res['max_rms_energy_s']:.2f}" if res['max_rms_energy_s'] is not None else "N/A"
            dtw = f"{res['dtw_cost']:.2f}"
            print(f"{res['song_folder']:<40} {gt:<18} {mc:<18} {mr:<18} {mrms:<15} {dtw:<10}")


if __name__ == "__main__":
    main()