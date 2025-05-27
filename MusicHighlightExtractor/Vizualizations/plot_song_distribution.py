# --- IMPORTS and CONSTANTS ---
import torch
import torch.nn as nn
import numpy as np
import librosa
import os
import random
import argparse
import matplotlib.pyplot as plt # Import main plotting library
import traceback
import seaborn as sns # Still useful for potential future plot types or palettes
import pickle
from multiprocessing import Pool, cpu_count
import time
import re # For parsing potentially messy numbers
from collections import defaultdict # For detailed skip counts

# ANSI escape codes for bold text
BOLD = "\033[1m"
RESET = "\033[0m"

try:
    from scipy import stats
    _scipy_available = True
except ImportError:
    _scipy_available = False
    print(f"{BOLD}Warning: scipy not found. Linear regression line and MAE/RMSE cannot be calculated.{RESET}")
    print(f"{BOLD}         Install it using: pip install scipy{RESET}")

SAMPLE_RATE = 22050
SKIP_DTW = "SKIP_DTW"
SKIP_METADATA_FILE = "SKIP_METADATA_FILE"
SKIP_AUDIO_FILE = "SKIP_AUDIO_FILE"
SKIP_MISSING_KEYS = "SKIP_MISSING_KEYS"
SKIP_PARSE_ERROR = "SKIP_PARSE_ERROR"
SKIP_DURATION_INVALID = "SKIP_DURATION_INVALID"
SKIP_OTHER_EXCEPTION = "SKIP_OTHER_EXCEPTION"
# --- END IMPORTS and CONSTANTS ---


# --- collect_data_for_song function ---
def collect_data_for_song(args):
    """
    Helper function to collect duration, highlight start/end times,
    and preview duration for a single song, filtering by DTW cost.
    Designed to be used with multiprocessing.Pool.
    Returns:
        - Tuple (song_duration, start_sec, end_sec, preview_dur_sec) on success.
        - A SKIP_* constant string on failure, indicating the reason.
    """
    song_info, root_dir, plot_cache_dir, max_dtw_cost = args
    song_name, _ = os.path.splitext(song_info)
    item_path = os.path.join(root_dir, song_name)
    metadata_path = os.path.join(item_path, 'metadata.txt')
    audio_path = os.path.join(item_path, 'full_song.mp3')
    plot_cache_path = os.path.join(plot_cache_dir, f'{song_name}_plot_multi_cache.pkl')

    try:
        # --- File Existence Checks ---
        if not os.path.exists(metadata_path): return SKIP_METADATA_FILE
        if not os.path.exists(audio_path): return SKIP_AUDIO_FILE

        # --- Read Metadata ---
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta = {k.strip(): v.strip() for line in f if ': ' in line for k, v in [line.strip().split(': ', 1)]}

        # --- DTW Cost Filtering ---
        if 'DTW Cost' not in meta: return SKIP_MISSING_KEYS
        try:
            dtw_cost = float(meta['DTW Cost'])
            if dtw_cost > max_dtw_cost: return SKIP_DTW
        except ValueError: return SKIP_PARSE_ERROR

        # --- Check Cache ---
        if os.path.exists(plot_cache_path):
            try:
                with open(plot_cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    if isinstance(cached_data, tuple) and len(cached_data) == 4:
                        s_dur, s_start, s_end, s_prev_dur = cached_data
                        if all(isinstance(x, (int, float)) for x in cached_data):
                            return s_dur, s_start, s_end, s_prev_dur
            except Exception as e:
                print(f"{BOLD}Warning: Error loading plot multi cache {plot_cache_path}: {e}. Recalculating.{RESET}")

        # --- Get song duration ---
        song_duration = librosa.get_duration(path=audio_path)
        if song_duration is None or song_duration < 1.0: return SKIP_DURATION_INVALID

        # --- Extract Timestamps ---
        if 'Detected Timestamp' not in meta or 'Preview End Timestamp' not in meta:
             return SKIP_MISSING_KEYS

        try:
            start_match = re.search(r'\d+(\.\d+)?', meta['Detected Timestamp'])
            end_match = re.search(r'\d+(\.\d+)?', meta['Preview End Timestamp'])
            if not start_match or not end_match: return SKIP_PARSE_ERROR

            highlight_start_sec = float(start_match.group(0))
            highlight_end_sec = float(end_match.group(0))

            preview_duration_sec = None
            if 'Preview Duration' in meta:
                 preview_dur_match = re.search(r'\d+(\.\d+)?', meta['Preview Duration'])
                 if preview_dur_match:
                     preview_duration_sec = float(preview_dur_match.group(0))

        except ValueError: return SKIP_PARSE_ERROR

        # --- Validate and Clamp Timestamps ---
        highlight_start_sec = max(0.0, min(highlight_start_sec, song_duration))
        highlight_end_sec = max(highlight_start_sec, min(highlight_end_sec, song_duration))

        # --- Determine Preview Duration ---
        calculated_preview_dur = highlight_end_sec - highlight_start_sec
        if preview_duration_sec is not None and preview_duration_sec >= 0:
             final_preview_dur_sec = preview_duration_sec
        else:
            final_preview_dur_sec = max(0.0, calculated_preview_dur)

        # --- Save to Cache ---
        try:
            os.makedirs(plot_cache_dir, exist_ok=True)
            data_to_cache = (song_duration, highlight_start_sec, highlight_end_sec, final_preview_dur_sec)
            with open(plot_cache_path, 'wb') as f:
                pickle.dump(data_to_cache, f)
        except Exception as e:
            print(f"{BOLD}Warning: Could not save plot multi cache {plot_cache_path}: {e}{RESET}")

        return song_duration, highlight_start_sec, highlight_end_sec, final_preview_dur_sec

    except FileNotFoundError: return SKIP_AUDIO_FILE
    except Exception as e:
        print(f"{BOLD}--- Error processing {song_name} for multi-plot data collection ---{RESET}")
        print(f"{BOLD}Error Type: {type(e).__name__} - {e}{RESET}")
        print(f"{BOLD}--- Skipping {song_name} ---{RESET}")
        return SKIP_OTHER_EXCEPTION
# --- END collect_data_for_song ---


# --- collect_plot_data_parallel function ---
def collect_plot_data_parallel(dataset_folder, plot_cache_dir="plot_data_cache", max_dtw_cost=1200.0):
    """
    Collects song durations, highlight start/end times, and preview durations
    in parallel, filtering by DTW cost and tracking skip reasons.
    Returns: Tuple of lists (song_durations, highlight_starts, highlight_ends, preview_durations)
             and a dictionary skip_counts.
    """
    print(f"{BOLD}Collecting multi-plot data from {dataset_folder} using parallel processing...{RESET}")
    print(f"{BOLD}Applying filter: Including songs with DTW Cost <= {max_dtw_cost}{RESET}")
    multi_cache_dir = os.path.join(plot_cache_dir, "multi_data")
    os.makedirs(multi_cache_dir, exist_ok=True)

    all_items = os.listdir(dataset_folder)
    song_dirs = [item for item in all_items if os.path.isdir(os.path.join(dataset_folder, item))]
    print(f"{BOLD}Found {len(song_dirs)} potential song directories.{RESET}")

    tasks = [(song_name, dataset_folder, multi_cache_dir, max_dtw_cost) for song_name in song_dirs]

    song_durations = []
    highlight_starts = []
    highlight_ends = []
    preview_durations = []
    skip_counts = defaultdict(int)
    processed_count = 0
    start_time = time.time()
    num_workers = min(cpu_count(), 8)
    print(f"{BOLD}Using {num_workers} worker processes.{RESET}")

    try:
        with Pool(num_workers) as pool:
            results = pool.imap_unordered(collect_data_for_song, tasks, chunksize=max(1, len(tasks) // (num_workers * 4)))
            processed_tasks = 0
            for result in results:
                processed_tasks += 1
                if isinstance(result, tuple) and len(result) == 4:
                    s_dur, s_start, s_end, s_prev_dur = result
                    song_durations.append(s_dur)
                    highlight_starts.append(s_start)
                    highlight_ends.append(s_end)
                    preview_durations.append(s_prev_dur)
                    processed_count += 1
                elif isinstance(result, str) and result.startswith("SKIP_"):
                    skip_counts[result] += 1
                else:
                    skip_counts[SKIP_OTHER_EXCEPTION] += 1
                    print(f"{BOLD}Warning: Unexpected return value from worker: {result}{RESET}")

                if processed_tasks % 100 == 0 or processed_tasks == len(tasks):
                    print(f"{BOLD}  Processed {processed_tasks}/{len(tasks)} tasks...{RESET}", end='\r')
        if len(tasks) > 0 : print() # Newline after progress indicator finishes

    except Exception as e:
        print(f"{BOLD}\nAn error occurred during parallel data collection: {e}{RESET}")
        traceback.print_exc() # Standard traceback, not bolded by default.

    end_time = time.time()
    total_attempted = len(tasks)
    total_skipped = sum(skip_counts.values())
    skipped_dtw_count = skip_counts.get(SKIP_DTW, 0)
    skipped_other_count = total_skipped - skipped_dtw_count

    print(f"{BOLD}\nFinished multi-plot data collection.{RESET}")
    print(f"{BOLD}-------------------- SUMMARY --------------------{RESET}")
    print(f"{BOLD}Attempted to process: {total_attempted} potential songs{RESET}")
    print(f"{BOLD}Successfully processed (extracted data): {processed_count} songs{RESET}")
    print(f"{BOLD}Total Skipped: {total_skipped} songs{RESET}")
    print(f"{BOLD}  - Skipped due to DTW Cost > {max_dtw_cost:.2f}: {skipped_dtw_count}{RESET}")
    print(f"{BOLD}  - Skipped due to other reasons: {skipped_other_count}{RESET}")
    if skipped_other_count > 0:
        print(f"{BOLD}    Detailed 'Other' Skip Reasons:{RESET}")
        for reason, count in sorted(skip_counts.items()): # Sort for consistent order
            if reason != SKIP_DTW:
                 print(f"{BOLD}      - {reason}: {count}{RESET}")
    print(f"{BOLD}Total time: {end_time - start_time:.2f} seconds.{RESET}")
    print(f"{BOLD}-----------------------------------------------{RESET}")

    return (song_durations, highlight_starts, highlight_ends, preview_durations), skip_counts
# --- END collect_plot_data_parallel ---


# --- Modified Generic Plotting Function ---
def plot_single_relationship(x_data, y_data, x_label, y_label, title, output_path,
                             plot_type='scatter', color_data=None, color_label=None,
                             add_regression=True):
    """
    Generates a single plot for a given relationship (X vs Y).
    Calculates and adds Linear Regression + MAE/RMSE for scatter plots IFF add_regression is True.
    Optionally colors scatter points based on color_data. Uses rcParams for font styles.
    """
    if not x_data or not y_data or len(x_data) != len(y_data):
        print(f"{BOLD}Warning: No valid/consistent data for plot '{title}'. Skipping.{RESET}")
        return

    print(f"{BOLD}\nGenerating plot: {title}{RESET}")
    print(f"{BOLD}  Output path: {output_path}{RESET}")
    if not add_regression and plot_type == 'scatter':
        print(f"{BOLD}  (Linear regression disabled for this plot){RESET}")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    x_np = np.array(x_data)
    y_np = np.array(y_data)
    color_np = np.array(color_data) if color_data is not None else None
    if color_np is not None and len(color_np) != len(x_np):
        print(f"{BOLD}Warning: Mismatch between data points ({len(x_np)}) and color data ({len(color_np)}) for plot '{title}'. Ignoring color.{RESET}")
        color_np = None

    fig, ax = plt.subplots(figsize=(14, 10))

    if plot_type == 'scatter':
        scatter_plot = ax.scatter(x_np, y_np, alpha=0.4, s=12, label='Data Points', c=color_np, cmap='viridis' if color_np is not None else None)
        if color_np is not None:
             cbar = fig.colorbar(scatter_plot, ax=ax)
             # fontweight for colorbar label set via rcParams or manually if needed
             cbar.ax.set_ylabel(color_label if color_label else 'Color Value', rotation=270, labelpad=25, fontweight='bold')


        if add_regression:
            regr_label_base = 'Linear Regression'
            if _scipy_available and len(x_np) > 1:
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_np, y_np)
                    predicted_y = slope * x_np + intercept
                    mae = np.mean(np.abs(y_np - predicted_y))
                    rmse = np.sqrt(np.mean((y_np - predicted_y)**2))
                    r_squared = r_value**2

                    print(f"{BOLD}  Linear Regression for '{y_label}' vs '{x_label}':{RESET}")
                    print(f"{BOLD}    Equation: y = {slope:.4f} * x + {intercept:.4f}{RESET}")
                    print(f"{BOLD}    MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r_squared:.4f}{RESET}")

                    line_x = np.array([x_np.min(), x_np.max()])
                    line_y = slope * line_x + intercept
                    regr_label = f'{regr_label_base}\ny = {slope:.2f}x + {intercept:.2f}\n(R²={r_squared:.3f}, MAE={mae:.2f})'
                    ax.plot(line_x, line_y, color='red', linestyle='--', linewidth=3, label=regr_label)
                    ax.legend(loc='best') # Legend text style handled by rcParams

                except Exception as e:
                    print(f"{BOLD}  Warning: Could not compute/plot linear regression: {e}{RESET}")
            elif not _scipy_available:
                 print(f"{BOLD}  Scipy not available, skipping linear regression ({regr_label_base}).{RESET}")
                 # Optionally add a placeholder label if you still want it in legend
                 # ax.plot([], [], color='red', linestyle='--', linewidth=3, label=f'{regr_label_base} (scipy unavailable)')
                 # ax.legend(loc='best')
            else:
                print(f"{BOLD}  Only one data point, cannot calculate regression.{RESET}")

    elif plot_type == 'hexbin':
         if color_np is not None: print(f"{BOLD}Warning: Color data ignored for hexbin plot.{RESET}")
         hb = ax.hexbin(x_np, y_np, gridsize=50, cmap='viridis', mincnt=1)
         cbar = fig.colorbar(hb, ax=ax)
         cbar.ax.set_ylabel('Number of Songs', rotation=270, labelpad=25, fontweight='bold')

    elif plot_type == 'hist2d':
        if color_np is not None: print(f"{BOLD}Warning: Color data ignored for hist2d plot.{RESET}")
        h, _, _, image = ax.hist2d(x_np, y_np, bins=50, cmap='viridis')
        cbar = fig.colorbar(image, ax=ax)
        cbar.ax.set_ylabel('Number of Songs', rotation=270, labelpad=25, fontweight='bold')

    else:
        print(f"{BOLD}Error: Unknown plot type '{plot_type}' for plot '{title}'. Skipping.{RESET}")
        plt.close(fig)
        return

    # Axis labels and title fontweights are handled by rcParams
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, pad=25)
    ax.grid(True, linestyle='--', alpha=0.6)

    fig.tight_layout()

    try:
        fig.savefig(output_path, dpi=150)
        print(f"{BOLD}  Plot saved successfully (DPI=150).{RESET}")
    except Exception as e:
        print(f"{BOLD}  Error saving plot '{output_path}': {e}{RESET}")
    finally:
        plt.close(fig)
# --- END plot_single_relationship ---


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates multiple plots with optional regression, filtering by DTW cost, reporting skips, and using larger, bold fonts.")
    parser.add_argument('--dataset_folder', type=str, required=True, help="Path to the dataset folder")
    parser.add_argument('--output_prefix', type=str, default='plots/song_analysis', help="Prefix for the output plot filenames")
    parser.add_argument('--plot_cache_dir', type=str, default='plot_data_cache', help="Base directory to cache collected multi-plot data")
    parser.add_argument('--max_dtw_cost', type=float, default=1200.0, help="Maximum DTW cost allowed (inclusive) from metadata.txt.")

    args = parser.parse_args()

    if not os.path.isdir(args.dataset_folder):
        print(f"{BOLD}Error: Dataset folder not found at {args.dataset_folder}{RESET}")
    else:
        output_dir = os.path.dirname(args.output_prefix)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        collected_data, skip_counts = collect_plot_data_parallel(
            args.dataset_folder,
            args.plot_cache_dir,
            args.max_dtw_cost
        )
        song_durations, highlight_starts, highlight_ends, preview_durations = collected_data

        if not song_durations:
            print(f"{BOLD}\nNo data collected after filtering. Cannot generate plots or calculate duration stats.{RESET}")
        else:
            # --- Set EVEN LARGER Font Sizes and BOLD using rcParams ---
            print(f"{BOLD}\nApplying EVEN LARGER and BOLD font styles for plots...{RESET}")
            plt.rcParams.update({
                'font.size': 18,             # Base font size
                'axes.labelsize': 20,        # X/Y axis labels
                'axes.titlesize': 22,        # Plot title
                'xtick.labelsize': 18,       # X tick labels
                'ytick.labelsize': 18,       # Y tick labels
                'legend.fontsize': 16,       # Legend text
                'legend.title_fontsize': 17, # Legend title
                'figure.titlesize': 24,      # Figure super title

                'font.weight': 'bold',       # Global font weight
                'axes.labelweight': 'bold',  # Weight for x and y labels
                'axes.titleweight': 'bold',  # Weight for axes titles
                'figure.titleweight': 'bold' # Weight for figure suptitle (if used)
                # Tick labels and legend text should inherit 'font.weight: bold'
            })
            # --- End Font Size and Weight Settings ---

            num_songs = len(song_durations)
            filter_str = f"DTW Cost <= {args.max_dtw_cost:.2f}"
            base_title = f"({num_songs} Songs, Filter: {filter_str})"

            # --- Preview Duration Stats ---
            print(f"{BOLD}\n--- Preview Duration Analysis ({num_songs} Processed Songs) ---{RESET}")
            count_within_range = 0
            count_outside_range = 0
            target_min = 29.0
            target_max = 31.0
            for dur in preview_durations:
                if target_min <= dur <= target_max:
                    count_within_range += 1
                else:
                    count_outside_range += 1
            print(f"{BOLD}Preview durations within [{target_min:.1f}s, {target_max:.1f}s]: {count_within_range}{RESET}")
            print(f"{BOLD}Preview durations outside [{target_min:.1f}s, {target_max:.1f}s]: {count_outside_range}{RESET}")
            if num_songs > 0:
                 percent_within = (count_within_range / num_songs) * 100
                 print(f"{BOLD}Percentage within range: {percent_within:.2f}%{RESET}")
            print(f"{BOLD}----------------------------------------------------{RESET}")
            # --- End Preview Duration Stats ---


            # --- Plotting Calls ---
            plot_single_relationship(
                x_data=song_durations, y_data=highlight_starts,
                x_label="Song Duration (seconds)", y_label="Highlight Start Timestamp (seconds)",
                title=f"Highlight Start vs. Song Duration {base_title}",
                output_path=f"{args.output_prefix}_start_regr.png",
                add_regression=True
            )
            plot_single_relationship(
                x_data=song_durations, y_data=highlight_ends,
                x_label="Song Duration (seconds)", y_label="Highlight End Timestamp (seconds)",
                title=f"Highlight End vs. Song Duration {base_title}",
                output_path=f"{args.output_prefix}_end_regr.png",
                add_regression=True
            )
            plot_single_relationship(
                x_data=song_durations, y_data=preview_durations,
                x_label="Song Duration (seconds)", y_label="Preview Duration (seconds)",
                title=f"Preview Duration vs. Song Duration {base_title}",
                output_path=f"{args.output_prefix}_preview_duration.png",
                add_regression=False # Regression might not be meaningful here
            )
            plot_single_relationship(
                x_data=song_durations, y_data=preview_durations,
                x_label="Song Duration (seconds)", y_label="Preview Duration (seconds)",
                title=f"Preview Duration vs. Song Duration (Colored by Start Time) {base_title}",
                output_path=f"{args.output_prefix}_preview_duration_colored_by_start.png",
                color_data=highlight_starts, color_label="Highlight Start Time (s)",
                add_regression=False # Regression might not be meaningful here
            )
            # --- End Plotting Calls ---

            print(f"{BOLD}\nAll plotting tasks complete.{RESET}")

# --- END Main Execution ---