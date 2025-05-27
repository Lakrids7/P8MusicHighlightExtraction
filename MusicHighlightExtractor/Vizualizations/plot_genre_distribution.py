# --- IMPORTS and CONSTANTS ---
import os
import argparse
import matplotlib.pyplot as plt # Import main plotting library
import traceback
import seaborn as sns
import pickle
from multiprocessing import Pool, cpu_count
import time
from collections import Counter, defaultdict # Added defaultdict
import re

try:
    from wordcloud import WordCloud
    _wordcloud_available = True
except ImportError:
    _wordcloud_available = False
    print("Warning: wordcloud library not found. Word cloud plot type disabled.")
    print("         Install it using: pip install wordcloud")

try:
    import squarify
    _squarify_available = True
except ImportError:
    _squarify_available = False
    print("Warning: squarify library not found. Treemap plot type disabled.")
    print("         Install it using: pip install squarify")

DEFAULT_CACHE_DIR = "genre_data_cache_multi"
# Add skip reasons (can reuse from other script or define specific ones)
SKIP_DTW = "SKIP_DTW"
SKIP_METADATA_FILE = "SKIP_METADATA_FILE"
SKIP_AUDIO_FILE = "SKIP_AUDIO_FILE"
SKIP_MISSING_KEYS = "SKIP_MISSING_KEYS"
SKIP_PARSE_ERROR = "SKIP_PARSE_ERROR"
SKIP_OTHER_EXCEPTION = "SKIP_OTHER_EXCEPTION"
# --- END IMPORTS and CONSTANTS ---


# --- Helper function for multiprocessing ---
def collect_genres_for_song(args):
    """
    Helper function to collect a LIST of genres for a single song,
    filtering by DTW cost first. Returns list of genres or skip reason string.
    """
    song_dir_name, root_dir, cache_dir, max_dtw_cost = args
    item_path = os.path.join(root_dir, song_dir_name)
    metadata_path = os.path.join(item_path, 'metadata.txt')
    audio_path = os.path.join(item_path, 'full_song.mp3')
    cache_path = os.path.join(cache_dir, f'{song_dir_name}_genres_cache.pkl')

    try:
        # --- File Existence ---
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

        # --- Check Genre Cache ---
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    if isinstance(cached_data, list):
                        return cached_data
            except Exception as e:
                print(f"Warning: Error loading genre cache {cache_path}: {e}. Recalculating.")

        # --- Process Genre Information ---
        genre_key = 'Genres'
        genres_list = []
        if genre_key in meta:
            genre_string = meta[genre_key].strip()
            if genre_string:
                raw_genres = genre_string.split(',')
                genres_list = [ g.strip().lower().capitalize() for g in raw_genres if g.strip() ]

        # --- Save to cache ---
        try:
             os.makedirs(cache_dir, exist_ok=True)
             with open(cache_path, 'wb') as f:
                 pickle.dump(genres_list, f)
        except Exception as e:
             print(f"Warning: Could not save genre cache {cache_path}: {e}")

        return genres_list

    except FileNotFoundError: return SKIP_AUDIO_FILE # Should be caught earlier
    except Exception as e:
        print(f"--- Error processing {song_dir_name} for genre data collection ---")
        print(f"Error Type: {type(e).__name__} - {e}")
        return SKIP_OTHER_EXCEPTION
# --- END collect_genres_for_song ---


# --- Data Collection Function ---
def collect_all_genre_occurrences_parallel(dataset_folder, cache_dir=DEFAULT_CACHE_DIR, max_dtw_cost=1200.0):
    """
    Collects all genre occurrences from all songs in parallel, filtering by DTW cost,
    and tracks skip reasons.
    Returns flat list of genres and skip_counts dictionary.
    """
    print(f"Collecting genre data from {dataset_folder} using parallel processing...")
    print(f"Applying filter: Including songs with DTW Cost <= {max_dtw_cost}")
    genre_cache_dir = os.path.join(cache_dir, "genre_data") # Specific subdir
    os.makedirs(genre_cache_dir, exist_ok=True)

    all_items = os.listdir(dataset_folder)
    song_dirs = [item for item in all_items if os.path.isdir(os.path.join(dataset_folder, item))]
    print(f"Found {len(song_dirs)} potential song directories.")

    tasks = [(song_dir, dataset_folder, genre_cache_dir, max_dtw_cost) for song_dir in song_dirs]

    all_genre_occurrences = []
    processed_songs_count = 0
    skip_counts = defaultdict(int) # Initialize skip counter
    start_time = time.time()
    num_workers = min(cpu_count(), 8)
    print(f"Using {num_workers} worker processes.")

    try:
        with Pool(num_workers) as pool:
            results_iterator = pool.imap_unordered(collect_genres_for_song, tasks, chunksize=max(1, len(tasks) // (num_workers * 4)))
            processed_tasks = 0
            for result in results_iterator:
                processed_tasks += 1
                if isinstance(result, list): # Success returns a list
                    processed_songs_count += 1
                    if result: # Only extend if the list isn't empty
                        all_genre_occurrences.extend(result)
                elif isinstance(result, str) and result.startswith("SKIP_"): # Skip returns string
                    skip_counts[result] += 1
                else: # Handle unexpected
                    skip_counts[SKIP_OTHER_EXCEPTION] += 1
                    print(f"Warning: Unexpected return value from genre worker: {result}")

                if processed_tasks % 100 == 0 or processed_tasks == len(tasks):
                    print(f"  Processed {processed_tasks}/{len(tasks)} tasks...", end='\r')

    except Exception as e:
        print(f"\nAn error occurred during parallel data collection: {e}")
        traceback.print_exc()

    end_time = time.time()
    total_attempted = len(tasks)
    total_skipped = sum(skip_counts.values())
    skipped_dtw_count = skip_counts.get(SKIP_DTW, 0)
    skipped_other_count = total_skipped - skipped_dtw_count

    print(f"\nFinished genre data collection.")
    print(f"-------------------- GENRE SUMMARY --------------------")
    print(f"Attempted to process: {total_attempted} potential songs")
    print(f"Successfully processed (metadata read & passed filters): {processed_songs_count} songs")
    print(f"Total Skipped: {total_skipped} songs")
    print(f"  - Skipped due to DTW Cost > {max_dtw_cost:.2f}: {skipped_dtw_count}")
    print(f"  - Skipped due to other reasons: {skipped_other_count}")
    if skipped_other_count > 0:
        print("    Detailed 'Other' Skip Reasons:")
        for reason, count in sorted(skip_counts.items()):
            if reason != SKIP_DTW:
                 print(f"      - {reason}: {count}")
    print(f"Collected a total of {len(all_genre_occurrences)} genre occurrences (tags) from processed songs.")
    print(f"Total time: {end_time - start_time:.2f} seconds.")
    print(f"-----------------------------------------------------")

    return all_genre_occurrences, skip_counts
# --- END collect_all_genre_occurrences_parallel ---


# --- Unified Plotting Function ---
def plot_genre_distribution(genres, output_prefix="genre_distribution", plot_type="bar_top_n", top_n=30, max_dtw_cost=None):
    """
    Plots the distribution of song genres using various methods. Uses larger fonts.
    """
    if not genres:
        print("No genre occurrences data provided for plotting. Cannot generate plot.")
        return

    output_dir = os.path.dirname(output_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    genre_counts = Counter(genres)
    num_unique_genres = len(genre_counts)
    total_genre_occurrences = len(genres)
    print(f"\nFound {num_unique_genres} unique genre tags across {total_genre_occurrences} total occurrences.")

    if num_unique_genres == 0:
        print("No unique genres found after processing. Cannot plot.")
        return

    actual_top_n = min(top_n, num_unique_genres)

    title_base = f"Song Genre Occurrence Distribution"
    if max_dtw_cost is not None:
        title_filter = f"DTW Cost <= {max_dtw_cost:.2f}"
    else:
        title_filter = "All Songs"

    # plt.style.use('seaborn-v0_8-darkgrid') # Style can be overridden by rcParams

    # === Plot Type 1: Bar Chart ===
    if plot_type == 'bar_top_n':
        print(f"Generating Top {actual_top_n} Genres Bar Chart...")
        sorted_genres = genre_counts.most_common(actual_top_n)
        if not sorted_genres: print("No genres to plot."); return

        plot_labels = [genre for genre, count in sorted_genres]
        plot_counts = [count for genre, count in sorted_genres]

        if num_unique_genres > actual_top_n:
            other_count = sum(count for genre, count in genre_counts.most_common()[actual_top_n:])
            plot_labels.append(f"Other ({num_unique_genres - actual_top_n} genres)")
            plot_counts.append(other_count)

        # Increase figure height based on number of labels and larger font
        fig_height = max(8, len(plot_labels) * 0.5) # Increased multiplier
        fig_width = 15 # Increased width
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        sns.barplot(x=plot_counts, y=plot_labels, palette="viridis", ax=ax, orient='h')

        ax.set_xlabel("Number of Times Genre Appears (Occurrences)")
        ax.set_ylabel("Genre")
        other_str = f' + Other' if num_unique_genres > actual_top_n else ''
        title = f"{title_base} (Top {actual_top_n}{other_str})\nFiltered by: {title_filter}"
        ax.set_title(title, pad=25) # Added padding

        max_count = max(plot_counts) if plot_counts else 1
        # Add count annotations with larger font size
        annotation_fontsize = plt.rcParams.get('xtick.labelsize', 14) # Base on tick size
        for index, value in enumerate(plot_counts):
             # Adjust horizontal offset based on max_count
             ax.text(value + (max_count * 0.015), index, str(value), va='center', fontsize=annotation_fontsize)

        fig.tight_layout() # Use fig.tight_layout with subplots
        output_file = f"{output_prefix}_bar_top_{actual_top_n}.png"

    # === Plot Type 2: Word Cloud ===
    elif plot_type == 'wordcloud':
        if not _wordcloud_available: print("WordCloud library not installed. Skipping."); return
        print("Generating Genre Word Cloud...")
        frequencies = dict(genre_counts)
        if not frequencies: print("No genre frequencies for word cloud."); return

        # Note: Word sizes within the cloud are relative to frequency.
        # We increase title size via rcParams.
        wordcloud = WordCloud(width=1600, height=1000, # Increased dimensions
                              background_color='white', colormap='viridis',
                              max_words=200, contour_width=1,
                              contour_color='steelblue', random_state=42
                             ).generate_from_frequencies(frequencies)

        fig, ax = plt.subplots(figsize=(18, 12)) # Increased size
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        title = f"{title_base} (Word Cloud)\nFiltered by: {title_filter}"
        ax.set_title(title, pad=25) # Use rcParams size, add padding
        fig.tight_layout(pad=0)
        output_file = f"{output_prefix}_wordcloud.png"

    # === Plot Type 3: Treemap ===
    elif plot_type == 'treemap':
        if not _squarify_available: print("squarify library not installed. Skipping."); return
        print(f"Generating Genre Treemap (Top {actual_top_n})...")
        sorted_genres = genre_counts.most_common(actual_top_n)
        if not sorted_genres: print("No genres to plot in treemap."); return

        plot_labels = [f"{genre}\n({count})" for genre, count in sorted_genres]
        plot_sizes = [count for genre, count in sorted_genres]
        colors = plt.cm.viridis([i/float(len(plot_labels)) for i in range(len(plot_labels))])

        fig, ax = plt.subplots(figsize=(16, 12)) # Increased size

        # Increase text size within the treemap boxes significantly
        squarify.plot(ax=ax, sizes=plot_sizes, label=plot_labels,
                      color=colors, alpha=0.8,
                      text_kwargs={'fontsize': 16, 'color': 'white'}) # Increased fontsize

        title = f"{title_base} (Treemap - Top {actual_top_n})\nFiltered by: {title_filter}"
        ax.set_title(title, pad=25) # Use rcParams size, add padding
        ax.axis('off')
        fig.tight_layout()
        output_file = f"{output_prefix}_treemap_top_{actual_top_n}.png"

    else:
        print(f"Error: Unknown plot type '{plot_type}'.")
        return

    try:
        fig.savefig(output_file, dpi=150) # Use fig.savefig
        print(f"Plot saved successfully to {output_file} (DPI=150)")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close(fig) # Close the figure
# --- END plot_genre_distribution ---


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot distribution of song genre occurrences using various methods, with large fonts.")
    parser.add_argument('--dataset_folder', type=str, required=True, help="Path to the dataset folder")
    parser.add_argument('--output_prefix', type=str, default='plots/genre_distribution', help="Base path and filename prefix for the output plot(s).")
    parser.add_argument('--cache_dir', type=str, default=DEFAULT_CACHE_DIR, help="Directory to cache collected genre data")
    parser.add_argument('--plot_type', type=str, default='bar_top_n', choices=['bar_top_n', 'wordcloud', 'treemap'], help="Type of plot.")
    parser.add_argument('--top_n', type=int, default=30, help="Number of top genres for 'bar_top_n' and 'treemap'.")
    parser.add_argument('--max_dtw_cost', type=float, default=1200.0, help="Maximum DTW cost allowed.")

    args = parser.parse_args()

    if args.top_n <= 0: args.top_n = 30

    if not os.path.isdir(args.dataset_folder):
        print(f"Error: Dataset folder not found at {args.dataset_folder}")
    else:
        output_dir = os.path.dirname(args.output_prefix)
        if output_dir: os.makedirs(output_dir, exist_ok=True)

        # Collect genres and skip counts
        all_genres, skip_counts = collect_all_genre_occurrences_parallel(
            args.dataset_folder,
            args.cache_dir,
            args.max_dtw_cost
        )

        if all_genres:
             # --- Set EVEN LARGER Font Sizes using rcParams ---
            print("\nApplying EVEN LARGER font sizes for plots...")
            plt.rcParams.update({
                'font.size': 18,             # Base font size
                'axes.labelsize': 20,        # X/Y axis labels
                'axes.titlesize': 22,        # Plot title
                'xtick.labelsize': 18,       # X tick labels
                'ytick.labelsize': 18,       # Y tick labels
                'legend.fontsize': 16,       # Legend text
                'legend.title_fontsize': 17, # Legend title
                'figure.titlesize': 24       # Figure super title
            })
            # --- End Font Size Settings ---

            # Generate the chosen plot
            plot_genre_distribution(
                all_genres,
                args.output_prefix,
                args.plot_type,
                args.top_n,
                args.max_dtw_cost
            )
            print("\nGenre plotting task complete.")
        else:
            print("\nNo genre data collected after filtering. Cannot generate genre plot.")

# --- END Main Execution ---