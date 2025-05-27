import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import traceback
import argparse
import librosa
import os
import re # For sanitizing filenames

# --- Constants (Defaults based on user specification) ---
SPECIFIED_SAMPLE_RATE = 22050
SPECIFIED_N_MELS = 128
SPECIFIED_N_FFT = 2048       # FFT window length
SPECIFIED_HOP_LENGTH = 512   # Hop size
SPECIFIED_WINDOW = 'hamming' # Window type
TARGET_FILENAMES = ["full_song.mp3", "original_preview.mp3"] # Files to look for

# --- Data Transformation (Copied from your training script) ---
class NormalizeSpectrogram(nn.Module):
    def forward(self, mel_db):
        if not isinstance(mel_db, torch.Tensor):
            mel_db = torch.tensor(mel_db, dtype=torch.float32)
        mean = mel_db.mean()
        std = mel_db.std()
        if std == 0:
            return mel_db - mean
        return (mel_db - mean) / (std + 1e-9)

# --- Function to sanitize strings for use in filenames ---
def sanitize_filename(name_string):
    """Removes or replaces characters unsafe for filenames."""
    # Remove potentially problematic characters
    sanitized = re.sub(r'[\\/*?:"<>|]', "", name_string)
    # Replace spaces with underscores (optional, but common)
    sanitized = sanitized.replace(" ", "_")
    # Limit length if necessary (optional)
    # max_len = 100
    # sanitized = sanitized[:max_len]
    return sanitized

# --- Function to plot a single spectrogram tensor ---
def plot_spectrogram(spec_tensor, title="Mel Spectrogram", time_per_frame_ms=23.2):
    """Plots a 2D spectrogram tensor."""
    # (No changes needed in the plotting logic itself)
    if not isinstance(spec_tensor, torch.Tensor):
        print(f"Warning: Expected a Tensor, got {type(spec_tensor)}. Skipping plot for '{title}'.")
        return False
    if spec_tensor.ndim != 2:
        squeezed_tensor = spec_tensor.squeeze()
        if squeezed_tensor.ndim == 2:
             print(f"Note: Original tensor shape was {spec_tensor.shape}, plotting squeezed version.")
             spec_tensor = squeezed_tensor
        else:
             print(f"Warning: Expected a 2D tensor for plotting, got shape {spec_tensor.shape}. Skipping plot for '{title}'.")
             return False
    if spec_tensor.numel() == 0:
        print(f"Warning: Tensor is empty. Skipping plot for '{title}'.")
        return False

    plt.figure(figsize=(12, 5))
    plt.imshow(spec_tensor.cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='dB (normalized)')
    plt.xlabel(f"Time Frames (~{time_per_frame_ms:.1f}ms per frame)")
    plt.ylabel(f"Mel Frequency Bins ({spec_tensor.shape[0]})")
    plt.title(title) # Title is passed in
    plt.tight_layout()
    return True

# --- Function to process a single audio file ---
# <<< Added 'folder_name' argument >>>
def process_and_visualize_audio(audio_path, folder_name, args, normalizer):
    """Loads audio, converts to mel spectrogram, normalizes, and plots."""
    print("-" * 20)
    print(f"Processing audio file: {audio_path.name} (from folder: {folder_name})")

    try:
        # --- Load Audio ---
        print("  Loading audio...")
        y, sr = librosa.load(str(audio_path), sr=args.sr, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"  Audio loaded. Duration: {duration:.2f} seconds.")

        # --- Calculate Mel Spectrogram ---
        print("  Calculating Mel spectrogram...")
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=args.n_mels, n_fft=args.n_fft,
            hop_length=args.hop_length, window=args.window
        )
        print("  Converting to dB scale...")
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # --- Convert to Tensor and Normalize ---
        print("  Converting to Tensor...")
        mel_db_tensor = torch.tensor(mel_db, dtype=torch.float32)
        print("  Normalizing spectrogram...")
        normalized_mel_tensor = normalizer(mel_db_tensor)
        print(f"  Final tensor shape: {normalized_mel_tensor.shape}")

        # --- Plotting ---
        print("  Generating plot...")
        time_per_frame_ms = (args.hop_length / args.sr) * 1000

        # <<< Construct title using folder_name and file stem >>>
        file_type_indicator = audio_path.stem # e.g., "full_song" or "original_preview"
        plot_title = f"Mel Spectrogram: {folder_name} ({file_type_indicator})"

        plotted_successfully = plot_spectrogram(
            normalized_mel_tensor,
            title=plot_title,
            time_per_frame_ms=time_per_frame_ms
        )

        # --- Save or Handle Plot ---
        if plotted_successfully:
            if args.save:
                output_folder_path = Path(args.output_folder)
                output_folder_path.mkdir(parents=True, exist_ok=True)

                # <<< Construct save filename using sanitized folder_name and file stem >>>
                safe_folder_name = sanitize_filename(folder_name)
                safe_file_stem = sanitize_filename(file_type_indicator)
                output_filename = f"{safe_folder_name}_{safe_file_stem}_mel_visualization.png"
                output_path = output_folder_path / output_filename

                try:
                    plt.savefig(output_path)
                    print(f"  Plot saved to {output_path}")
                    plt.close()
                except Exception as save_err:
                    print(f"  Error saving plot to {output_path}: {save_err}")
                    plt.close()
            return True
        else:
            print("  Spectrogram could not be plotted.")
            return False

    except FileNotFoundError:
        print(f"  Error: Could not find audio file '{audio_path.name}' during processing.")
        return False
    except Exception as e:
        print(f"\n  --- An error occurred processing {audio_path.name}: ---")
        print(f"  Error Type: {type(e).__name__}")
        print(f"  Error Message: {e}")
        print("  --- Traceback ---")
        traceback.print_exc()
        print("  -----------------")
        if plt.fignum_exists(plt.gcf().number):
             plt.close()
        return False

# --- Main Execution Logic ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    # (Parser setup remains the same)
    parser = argparse.ArgumentParser(description="Find specific audio files in a folder, convert to Mel spectrogram, and visualize.")
    parser.add_argument('--input_folder', type=str, required=True, help="Path to the folder containing audio files.")
    parser.add_argument('--sr', type=int, default=SPECIFIED_SAMPLE_RATE, help=f"Sample rate (default: {SPECIFIED_SAMPLE_RATE})")
    parser.add_argument('--n_mels', type=int, default=SPECIFIED_N_MELS, help=f"Number of Mel bins (default: {SPECIFIED_N_MELS})")
    parser.add_argument('--n_fft', type=int, default=SPECIFIED_N_FFT, help=f"FFT window size (default: {SPECIFIED_N_FFT})")
    parser.add_argument('--hop_length', type=int, default=SPECIFIED_HOP_LENGTH, help=f"Hop length (default: {SPECIFIED_HOP_LENGTH})")
    parser.add_argument('--window', type=str, default=SPECIFIED_WINDOW, help=f"Window function (default: '{SPECIFIED_WINDOW}')")
    parser.add_argument('--save', action='store_true', default=False, help="Save plots to files instead of displaying.")
    parser.add_argument('--no-save', action='store_false', dest='save', help="Display plots interactively (default).")
    parser.add_argument('--output_folder', type=str, default="spectrogram_visualizations", help="Directory to save plots if --save is used.")
    args = parser.parse_args()

    # --- Validate Input Folder ---
    input_folder_path = Path(args.input_folder)
    if not input_folder_path.is_dir():
        print(f"Error: Input folder not found or not a directory: '{args.input_folder}'")
        exit()

    # <<< Get the folder name >>>
    song_folder_name = input_folder_path.name

    print(f"Processing folder: {song_folder_name} (Path: {input_folder_path})")
    print(f"Looking for {TARGET_FILENAMES}")
    print(f"Using Parameters: SR={args.sr}, N_Mels={args.n_mels}, N_FFT={args.n_fft}, Hop Length={args.hop_length}, Window='{args.window}'")

    normalizer = NormalizeSpectrogram()
    files_processed_count = 0
    plots_generated_count = 0

    # --- Iterate through target filenames and process if found ---
    for filename in TARGET_FILENAMES:
        audio_path = input_folder_path / filename
        if audio_path.is_file():
            files_processed_count += 1
            # <<< Pass song_folder_name to the processing function >>>
            if process_and_visualize_audio(audio_path, song_folder_name, args, normalizer):
                plots_generated_count += 1
        else:
            print(f"\nFile not found, skipping: {audio_path.name}")

    print("-" * 20)

    # --- Final Actions ---
    # (Final summary logic remains the same)
    if files_processed_count == 0:
        print(f"Error: Neither '{TARGET_FILENAMES[0]}' nor '{TARGET_FILENAMES[1]}' were found in '{input_folder_path}'.")
    else:
        if plots_generated_count == 0:
             print(f"Processed {files_processed_count} audio file(s), but no spectrograms were successfully plotted (check for errors above).")
        elif args.save:
            print(f"\nFinished saving {plots_generated_count} plot(s) to {args.output_folder}")
        else:
            print(f"\nDisplaying {plots_generated_count} generated plot(s)...")
            plt.show()

    print("\nProcessing finished.")