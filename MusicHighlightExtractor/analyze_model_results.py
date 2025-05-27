import os
import json
import math
import argparse
import re
import glob
import csv
import numpy as np
import pandas as pd # Added pandas
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

# --- Constants ---
N_MELS = 128
CHUNK_FRAMES = 128
TARGET_DIM = 2
MEL_SUFFIX = ".npy"
AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
SR_TARGET_FOR_DTW = 22050
N_FFT_DTW = 2048
HOP_LENGTH_DTW = 512
N_MELS_DTW = 64
TOP_N_SONGS = 100 # For best/worst lists
FIGURES_SUBDIR = "summary_plots" # For generated summary plots

# --- Model Definition (Same as before) ---
def positional_encoding(L, D, device):
    pe = torch.zeros(L, D, device=device)
    pos = torch.arange(0, L, dtype=torch.float, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, D, 2, device=device) * (-math.log(10000.0) / D))
    pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe

class MusicHighlighter(nn.Module):
    def __init__(self, dim=64):
        super().__init__(); self.feat_dim = dim * 4
        def conv(ic, oc, k, s): return nn.Sequential(nn.Conv2d(ic, oc, k, s), nn.BatchNorm2d(oc), nn.ReLU())
        self.conv_blocks = nn.Sequential(conv(1, dim, (3, N_MELS), (2, 1)), conv(dim, dim*2, (4, 1), (2, 1)), conv(dim*2, self.feat_dim, (4, 1), (2, 1)))
        self.attn_mlp = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim), nn.ReLU(), nn.Dropout(0.5), nn.Linear(self.feat_dim, self.feat_dim), nn.Tanh(), nn.Dropout(0.5), nn.Linear(self.feat_dim, 1))
        self.regression_head = nn.Sequential(nn.Linear(self.feat_dim, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, TARGET_DIM), nn.Sigmoid())

    def forward(self, x, lengths):
        B, L_max, M_n_mels, T_chunk_frames = x.shape; dev = x.device
        x_permuted = x.permute(0, 1, 3, 2)
        x_reshaped = x_permuted.reshape(B * L_max, 1, T_chunk_frames, M_n_mels)
        chunk_features = torch.max(self.conv_blocks(x_reshaped).squeeze(3), dim=2)[0]
        h_t = chunk_features.view(B, L_max, -1)
        pe = positional_encoding(L_max, self.feat_dim, dev).unsqueeze(0)
        attn_input = (h_t + pe).view(B * L_max, -1)
        attn_logits = self.attn_mlp(attn_input).view(B, L_max)
        mask = torch.arange(L_max, device=dev)[None, :] >= lengths[:, None]
        attn_logits.masked_fill_(mask, -float('inf'))
        alpha_t = torch.softmax(attn_logits, dim=1)
        weighted_features = torch.bmm(alpha_t.unsqueeze(1), h_t).squeeze(1)
        prediction = self.regression_head(weighted_features)
        return prediction, alpha_t

# --- Helper function to parse metadata (Same as before) ---
def parse_metadata(meta_path):
    try:
        with open(meta_path, 'r', encoding='utf-8') as f: text = f.read()
        start_match = re.search(r"Detected Timestamp:\s*([\d.]+)", text)
        end_match = re.search(r"Preview End Timestamp:\s*([\d.]+)", text)
        total_dur_match = re.search(r"Full Song Duration:\s*([\d.]+)", text)
        if start_match and end_match and total_dur_match:
            start_time_abs = float(start_match.group(1)); end_time_abs = float(end_match.group(1))
            total_duration_s = float(total_dur_match.group(1))
            if total_duration_s > 0 and end_time_abs > start_time_abs:
                start_time_abs = max(0, min(start_time_abs, total_duration_s))
                end_time_abs = max(0, min(end_time_abs, total_duration_s))
                if end_time_abs <= start_time_abs : return None, None
                gt_highlight_norm = np.array([start_time_abs / total_duration_s, end_time_abs / total_duration_s], dtype=np.float32)
                return gt_highlight_norm, total_duration_s
    except Exception as e: print(f"Error parsing metadata {meta_path}: {e}")
    return None, None

# --- Dataset for Analysis (Same as before) ---
class AnalysisSongDataset(Dataset):
    def __init__(self, song_dirs):
        self.items = []
        for song_dir_path_loop_var in song_dirs:
            mel_path = os.path.join(song_dir_path_loop_var, f"full_song{MEL_SUFFIX}")
            meta_path = os.path.join(song_dir_path_loop_var, "metadata.txt")
            current_audio_path = None
            for ext in AUDIO_EXTENSIONS:
                potential_audio_path = os.path.join(song_dir_path_loop_var, f"full_song{ext}")
                if os.path.exists(potential_audio_path): current_audio_path = potential_audio_path; break
            if os.path.exists(mel_path) and os.path.exists(meta_path) and current_audio_path:
                self.items.append((mel_path, meta_path, current_audio_path, song_dir_path_loop_var))
            else:
                problem = [p for p, c in [("Mel", not os.path.exists(mel_path)), ("Meta", not os.path.exists(meta_path)), ("Audio", not current_audio_path)] if c]
                if problem: print(f"Warning: Skipping {song_dir_path_loop_var} due to missing: {', '.join(problem)}")
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        mel_path_item, meta_path_item, audio_path_item, song_dir_path_item = self.items[idx]
        try:
            mel_chunks = np.load(mel_path_item)
            gt_highlight_norm, total_duration_s = parse_metadata(meta_path_item)
            if gt_highlight_norm is not None and total_duration_s is not None and \
               mel_chunks.ndim == 3 and mel_chunks.shape[1:] == (N_MELS, CHUNK_FRAMES):
                return (torch.from_numpy(mel_chunks), torch.from_numpy(gt_highlight_norm),
                        total_duration_s, audio_path_item, os.path.basename(song_dir_path_item))
        except Exception as e: print(f"Error loading item {song_dir_path_item}: {e}")
        return None
def analysis_collate_fn(batch): return [item for item in batch if item is not None][0] if any(batch) else None

# --- Function to compute DTW cost (Same as before) ---
def compute_dtw_cost(audio1_segment, audio2_segment, sr):
    if audio1_segment.size == 0 or audio2_segment.size == 0: return np.nan
    try:
        mel1 = librosa.feature.melspectrogram(y=audio1_segment, sr=sr, n_fft=N_FFT_DTW, hop_length=HOP_LENGTH_DTW, n_mels=N_MELS_DTW)
        mel2 = librosa.feature.melspectrogram(y=audio2_segment, sr=sr, n_fft=N_FFT_DTW, hop_length=HOP_LENGTH_DTW, n_mels=N_MELS_DTW)
        log_mel1 = librosa.power_to_db(mel1, ref=np.max); log_mel2 = librosa.power_to_db(mel2, ref=np.max)
        D, _ = librosa.sequence.dtw(X=log_mel1, Y=log_mel2, metric='euclidean')
        return D[-1, -1]
    except Exception as e: print(f"Error computing DTW: {e}"); return np.nan

# --- Function to calculate attention within a window (Same as before) ---
def calculate_attention_in_window(attention_scores, chunk_center_times, window_start_s, window_end_s):
    if attention_scores is None or chunk_center_times is None or len(attention_scores) != len(chunk_center_times): return np.nan
    relevant_attention_sum = 0.0
    for i, center_time in enumerate(chunk_center_times):
        if window_start_s <= center_time < window_end_s: relevant_attention_sum += attention_scores[i]
    return relevant_attention_sum

# --- Helper for scatter plots ---
def add_identity_line(ax, color='red', linestyle='--'):
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, color=color, linestyle=linestyle, alpha=0.7, label='y=x (Ideal)')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

# --- Plotting Functions ---
def generate_and_save_statistics(df, output_path, dataset_name, num_processed_songs):
    stats_text = [f"Aggregate Performance Metrics for {num_processed_songs} songs from '{dataset_name}' set:\n"]
    metrics_to_summarize = {
        "error_avg_start_end_s": "Average Boundary Error (Start/End, s)", # Clarified
        "error_center_s": "Center Time Error (s)",
        "error_start_s": "Start Time Error (s)",
        "error_end_s": "End Time Error (s)",
        "error_duration_s": "Highlight Duration Error (s)", # Added
        "dtw_cost_mel": "DTW Mel Cost"
    }
    for col, name in metrics_to_summarize.items():
        if col in df.columns:
            series = df[col].dropna()
            if not series.empty:
                stats_text.append(f"\n--- {name} ---")
                stats_text.append(f"  Mean:   {series.mean():.3f}")
                stats_text.append(f"  Median: {series.median():.3f}")
                stats_text.append(f"  Std Dev:{series.std():.3f}")
                stats_text.append(f"  Min:    {series.min():.3f}")
                stats_text.append(f"  Max:    {series.max():.3f}")
            else:
                stats_text.append(f"\n--- {name} ---")
                stats_text.append(f"  No valid data to compute statistics.")
    
    try:
        with open(output_path, 'w') as f:
            f.write("\n".join(stats_text))
        print(f"Aggregate performance metrics saved to {output_path}")
    except IOError:
        print(f"Error: Could not write aggregate metrics to {output_path}")

def generate_histograms(df, plots_dir):
    print("Generating histograms...")
    # Avg Start/End Error
    plt.figure(figsize=(10, 6))
    df['error_avg_start_end_s'].dropna().hist(bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Average Boundary Error (Start/End)')
    plt.xlabel('Average Boundary Error (seconds)')
    plt.ylabel('Number of Songs')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "histogram_avg_boundary_error.png")); plt.close() # Renamed file for clarity

    # Center Time Error
    plt.figure(figsize=(10, 6))
    df['error_center_s'].dropna().hist(bins=30, color='lightcoral', edgecolor='black')
    plt.title('Distribution of Center Time Error')
    plt.xlabel('Center Error (seconds)')
    plt.ylabel('Number of Songs')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "histogram_center_error.png")); plt.close()

    # Duration Error (New)
    duration_errors = df['error_duration_s'].dropna()
    if not duration_errors.empty:
        plt.figure(figsize=(10, 6))
        duration_errors.hist(bins=30, color='gold', edgecolor='black')
        plt.title('Distribution of Highlight Duration Error')
        plt.xlabel('Duration Error (seconds)')
        plt.ylabel('Number of Songs')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "histogram_duration_error.png")); plt.close()
    else:
        print("Skipping duration error histogram: no valid data.")


    # DTW Cost
    dtw_costs = df['dtw_cost_mel'].dropna()
    if not dtw_costs.empty:
        plt.figure(figsize=(10, 6))
        dtw_costs.hist(bins=30, color='mediumseagreen', edgecolor='black')
        plt.title('Distribution of DTW Mel Cost')
        plt.xlabel('DTW Cost')
        plt.ylabel('Number of Songs')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "histogram_dtw_cost.png")); plt.close()
    else:
        print("Skipping DTW cost histogram: no valid DTW data.")
    print("Histograms saved.")


def generate_scatter_plots(df, plots_dir):
    print("Generating scatter plots...")
    # Start Times
    plt.figure(figsize=(8, 8))
    plt.scatter(df['gt_start_s_abs'], df['pred_start_s_abs'], alpha=0.6, edgecolors='w', linewidth=0.5)
    add_identity_line(plt.gca())
    plt.title('Predicted vs. Ground Truth Start Times')
    plt.xlabel('Ground Truth Start Time (s)')
    plt.ylabel('Predicted Start Time (s)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "scatter_start_times.png")); plt.close()

    # End Times
    plt.figure(figsize=(8, 8))
    plt.scatter(df['gt_end_s_abs'], df['pred_end_s_abs'], alpha=0.6, edgecolors='w', linewidth=0.5)
    add_identity_line(plt.gca())
    plt.title('Predicted vs. Ground Truth End Times')
    plt.xlabel('Ground Truth End Time (s)')
    plt.ylabel('Predicted End Time (s)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "scatter_end_times.png")); plt.close()

    # Center Times
    plt.figure(figsize=(8, 8))
    plt.scatter(df['gt_center_s_abs'], df['pred_center_s_abs'], alpha=0.6, edgecolors='w', linewidth=0.5)
    add_identity_line(plt.gca())
    plt.title('Predicted vs. Ground Truth Center Times')
    plt.xlabel('Ground Truth Center Time (s)')
    plt.ylabel('Predicted Center Time (s)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "scatter_center_times.png")); plt.close()
    
    # DTW Cost vs. Avg Error
    dtw_costs = df['dtw_cost_mel'].dropna()
    avg_errors_for_dtw = df.loc[dtw_costs.index, 'error_avg_start_end_s'].dropna()
    common_indices = dtw_costs.index.intersection(avg_errors_for_dtw.index)
    
    if not common_indices.empty:
        plt.figure(figsize=(10, 6))
        plt.scatter(df.loc[common_indices, 'error_avg_start_end_s'], df.loc[common_indices, 'dtw_cost_mel'], alpha=0.6, edgecolors='w', linewidth=0.5)
        plt.title('DTW Mel Cost vs. Average Boundary Error')
        plt.xlabel('Average Boundary Error (s)')
        plt.ylabel('DTW Mel Cost')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "scatter_dtw_vs_boundary_error.png")); plt.close() # Renamed file
    else:
        print("Skipping DTW vs Error scatter plot: insufficient common valid data.")
    print("Scatter plots saved.")

def generate_cumulative_error_plot(errors_list, plots_dir):
    print("Generating cumulative error plot (for Average Boundary Error)...")
    if not errors_list:
        print("Skipping cumulative error plot: no error data.")
        return
        
    errors_sorted = np.sort(np.array(errors_list))
    cumulative_percentage = np.arange(1, len(errors_sorted) + 1) / len(errors_sorted) * 100

    plt.figure(figsize=(10, 6))
    plt.plot(errors_sorted, cumulative_percentage, marker='.', linestyle='-', color='darkcyan')
    plt.title('Cumulative Distribution of Average Boundary Error (Start/End)')
    plt.xlabel('Average Boundary Error (seconds)')
    plt.ylabel('Percentage of Songs (%)')
    plt.grid(True, linestyle=':', alpha=0.7)
    # Add some key threshold lines
    for threshold in [1, 2, 3, 5, 10]: # Added 10s
        if errors_sorted[-1] >= threshold: # Only plot if threshold is within error range
            count_below_threshold = np.sum(errors_sorted <= threshold)
            percent_below = (count_below_threshold / len(errors_sorted)) * 100
            plt.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5)
            plt.text(threshold + 0.1, percent_below - 5 if percent_below > 10 else percent_below + 5, f'{percent_below:.1f}% $\leq$ {threshold}s', rotation=0, color='dimgray')
    plt.ylim(0, 101)
    plt.xlim(left=0)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "cumulative_boundary_error_distribution.png")); plt.close() # Renamed file
    print("Cumulative error plot saved.")


# --- Main Analysis Function ---
def analyze_songs(model_path, split_file_path, output_root_dir, device_str, max_songs_to_process=None):
    device = torch.device(device_str)
    model = MusicHighlighter().to(device); model.eval()
    try: model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e: print(f"Error loading model: {e}"); return

    dataset_name = ""
    song_dirs_to_process = []
    try:
        with open(split_file_path, 'r') as f: split_data = json.load(f)
        
        song_dirs_to_process = split_data.get("test", [])
        dataset_name = "test"
        if not song_dirs_to_process:
            print(f"Warning: 'test' set not found or is empty in {split_file_path}. Falling back to 'val' set.")
            song_dirs_to_process = split_data.get("val", [])
            dataset_name = "validation"
            if not song_dirs_to_process:
                print(f"Error: Neither 'test' nor 'val' sets found or are empty in {split_file_path}. Cannot proceed.")
                return
        
        if not song_dirs_to_process: 
             print(f"Error: No usable song directories found in '{dataset_name}' set from {split_file_path}.")
             return
        print(f"Using '{dataset_name}' set for analysis from {split_file_path}.")

    except Exception as e: print(f"Error loading split file {split_file_path}: {e}"); return

    dataset = AnalysisSongDataset(song_dirs_to_process)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=analysis_collate_fn)
    if len(dataset) == 0: print(f"No valid songs found in the '{dataset_name}' dataset."); return
    
    os.makedirs(output_root_dir, exist_ok=True)
    plots_output_dir = os.path.join(output_root_dir, FIGURES_SUBDIR)
    os.makedirs(plots_output_dir, exist_ok=True)
    
    all_song_results_data = []
    csv_header = [
        "song_id", "song_duration_s",
        "gt_start_s_abs", "gt_end_s_abs", "gt_center_s_abs", "gt_duration_s",
        "pred_start_s_abs", "pred_end_s_abs", "pred_center_s_abs", "pred_duration_s",
        "error_center_s", "error_start_s", "error_end_s", 
        "error_avg_start_end_s", "error_duration_s", # Added error_duration_s
        "dtw_cost_mel",
        "attention_in_pred_window", "attention_in_gt_window"
    ]

    num_to_process = len(dataset)
    if max_songs_to_process is not None and max_songs_to_process > 0 and max_songs_to_process < len(dataset):
        num_to_process = max_songs_to_process
        print(f"Analyzing the first {num_to_process} of {len(dataset)} valid songs from '{dataset_name}' set for metrics and detailed output...")
    else:
        print(f"Analyzing all {len(dataset)} valid songs from '{dataset_name}' set for metrics and detailed output...")

    total_attn_in_pred = 0; total_attn_in_gt = 0; valid_attn_entries = 0
    processed_count = 0

    for i, data_item in enumerate(dataloader):
        if max_songs_to_process is not None and max_songs_to_process > 0 and processed_count >= max_songs_to_process:
            break 

        if data_item is None: continue
        
        mel_chunks_tensor, gt_highlight_norm_tensor, \
        song_duration_s, audio_file_path, song_id = data_item

        if not (song_duration_s and song_duration_s > 0):
            print(f"Skipping {song_id}: invalid duration {song_duration_s}"); continue
            
        mel_for_model = mel_chunks_tensor.unsqueeze(0).to(device)
        num_chunks_val = mel_chunks_tensor.shape[0]
        num_chunks_tensor = torch.tensor([num_chunks_val], device=device)

        with torch.no_grad():
            pred_norm, attn_scores_tensor = model(mel_for_model, num_chunks_tensor)

        pred_norm_np = pred_norm.squeeze(0).cpu().numpy()
        gt_norm_np = gt_highlight_norm_tensor.cpu().numpy()
        attn_scores_np = attn_scores_tensor.squeeze(0)[:num_chunks_val].cpu().numpy()

        gt_start_s = gt_norm_np[0] * song_duration_s; gt_end_s = gt_norm_np[1] * song_duration_s
        pred_start_s = max(0, min(pred_norm_np[0] * song_duration_s, song_duration_s))
        pred_end_s = max(0, min(pred_norm_np[1] * song_duration_s, song_duration_s))
        if pred_end_s < pred_start_s: pred_end_s = min(pred_start_s + 0.1, song_duration_s) 

        gt_center_s = (gt_start_s + gt_end_s) / 2; pred_center_s = (pred_start_s + pred_end_s) / 2
        err_center = abs(pred_center_s - gt_center_s); err_start = abs(pred_start_s - gt_start_s)
        err_end = abs(pred_end_s - gt_end_s); err_avg_start_end = (err_start + err_end) / 2.0

        gt_duration_s_val = gt_end_s - gt_start_s
        pred_duration_s_val = pred_end_s - pred_start_s
        error_duration_s = abs(pred_duration_s_val - gt_duration_s_val)


        attention_in_pred_window = np.nan; attention_in_gt_window = np.nan; chunk_center_times = None
        if num_chunks_val > 0:
            actual_chunk_duration_s = song_duration_s / num_chunks_val
            chunk_start_times = np.arange(num_chunks_val) * actual_chunk_duration_s
            chunk_center_times = chunk_start_times + actual_chunk_duration_s / 2.0
            attention_in_pred_window = calculate_attention_in_window(attn_scores_np, chunk_center_times, pred_start_s, pred_end_s)
            attention_in_gt_window = calculate_attention_in_window(attn_scores_np, chunk_center_times, gt_start_s, gt_end_s)
            if not np.isnan(attention_in_pred_window) and not np.isnan(attention_in_gt_window):
                total_attn_in_pred += attention_in_pred_window
                total_attn_in_gt += attention_in_gt_window
                valid_attn_entries +=1
        
        song_out_dir = os.path.join(output_root_dir, song_id) 
        os.makedirs(song_out_dir, exist_ok=True)
        
        dtw_mel_cost = np.nan
        try:
            y_audio, sr_orig = librosa.load(audio_file_path, sr=None, mono=True)
            y_audio_dtw = librosa.resample(y_audio, orig_sr=sr_orig, target_sr=SR_TARGET_FOR_DTW) if sr_orig != SR_TARGET_FOR_DTW else y_audio
            sr_dtw = SR_TARGET_FOR_DTW if sr_orig != SR_TARGET_FOR_DTW else sr_orig
            def get_segment(y, sr, start_s, end_s):
                start_sample = max(0, int(start_s * sr)); end_sample = min(len(y), int(end_s * sr))
                return y[start_sample:end_sample] if end_sample > start_sample else np.array([])
            gt_segment_audio = get_segment(y_audio_dtw, sr_dtw, gt_start_s, gt_end_s)
            pred_segment_audio = get_segment(y_audio_dtw, sr_dtw, pred_start_s, pred_end_s)
            dtw_mel_cost = compute_dtw_cost(gt_segment_audio, pred_segment_audio, sr_dtw)
            sf.write(os.path.join(song_out_dir, "highlight_ground_truth.wav"), get_segment(y_audio, sr_orig, gt_start_s, gt_end_s), sr_orig)
            sf.write(os.path.join(song_out_dir, "highlight_predicted.wav"), get_segment(y_audio, sr_orig, pred_start_s, pred_end_s), sr_orig)
        except Exception as e: print(f"Audio/DTW error for {song_id}: {e}")

        try:
            time_ax_attn = chunk_center_times if chunk_center_times is not None else (np.arange(len(attn_scores_np)) * (song_duration_s / num_chunks_val) if num_chunks_val > 0 else np.array([]))
            if len(time_ax_attn) == len(attn_scores_np) and len(attn_scores_np) > 0:
                plt.figure(figsize=(12, 6))
                plt.plot(time_ax_attn, attn_scores_np, label='Attention Score', color='dodgerblue', lw=2)
                plt.axvspan(gt_start_s, gt_end_s, color='mediumseagreen', alpha=0.4, label=f'GT ({gt_start_s:.1f}-{gt_end_s:.1f}s)')
                plt.axvspan(pred_start_s, pred_end_s, color='salmon', alpha=0.4, label=f'Pred ({pred_start_s:.1f}-{pred_end_s:.1f}s)')
                plt.xlabel("Time (s)"); plt.ylabel("Attention Score"); plt.title(f"Attention for {song_id}")
                plt.legend(); plt.grid(True, linestyle=':', alpha=0.7); plt.tight_layout()
                plt.savefig(os.path.join(song_out_dir, "attention_scores.png")); plt.close()
            elif len(attn_scores_np) > 0: print(f"Warning: Could not plot attention for {song_id} due to time axis mismatch or empty scores.")
        except Exception as e: print(f"Plotting error for {song_id}: {e}")

        with open(os.path.join(song_out_dir, "prediction_error_info.txt"), 'w') as f:
            f.write(f"Song ID: {song_id}\nDuration: {song_duration_s:.2f}s\n"
                    f"GT Norm: [{gt_norm_np[0]:.4f}, {gt_norm_np[1]:.4f}], GT Sec: {gt_start_s:.2f}s-{gt_end_s:.2f}s (Center: {gt_center_s:.2f}s, Duration: {gt_duration_s_val:.2f}s)\n"
                    f"Pred Norm: [{pred_norm_np[0]:.4f}, {pred_norm_np[1]:.4f}], Pred Sec: {pred_start_s:.2f}s-{pred_end_s:.2f}s (Center: {pred_center_s:.2f}s, Duration: {pred_duration_s_val:.2f}s)\n"
                    f"Err Center: {err_center:.2f}s, Err Start: {err_start:.2f}s, Err End: {err_end:.2f}s, Err Avg Boundary: {err_avg_start_end:.2f}s, Err Duration: {error_duration_s:.2f}s\n"
                    f"DTW Mel Cost: {dtw_mel_cost:.4f}\n"
                    f"Attention in Predicted Window: {attention_in_pred_window:.4f}\n"
                    f"Attention in Ground Truth Window: {attention_in_gt_window:.4f}\n")

        all_song_results_data.append({
            "song_id": song_id, "song_duration_s": song_duration_s,
            "gt_start_s_abs": gt_start_s, "gt_end_s_abs": gt_end_s, "gt_center_s_abs": gt_center_s, "gt_duration_s": gt_duration_s_val,
            "pred_start_s_abs": pred_start_s, "pred_end_s_abs": pred_end_s, "pred_center_s_abs": pred_center_s, "pred_duration_s": pred_duration_s_val,
            "error_center_s": err_center, "error_start_s": err_start,
            "error_end_s": err_end, "error_avg_start_end_s": err_avg_start_end,
            "error_duration_s": error_duration_s, # Added
            "dtw_cost_mel": dtw_mel_cost,
            "attention_in_pred_window": attention_in_pred_window,
            "attention_in_gt_window": attention_in_gt_window
        })
        processed_count += 1
        print(f"Processed ({processed_count}/{num_to_process}): {song_id} | Avg Boundary Err: {err_avg_start_end:.2f}s | Duration Err: {error_duration_s:.2f}s | DTW: {dtw_mel_cost:.2f}")


    if not all_song_results_data:
        print("No songs were processed successfully. Exiting.")
        return

    results_df = pd.DataFrame(all_song_results_data)
    
    sorted_results_for_csv = sorted(
        all_song_results_data, 
        key=lambda x: (isinstance(x["error_avg_start_end_s"], float) and np.isnan(x["error_avg_start_end_s"]), x["error_avg_start_end_s"])
    )


    summary_csv_path = os.path.join(output_root_dir, f"analysis_summary_{dataset_name}_set_{processed_count}_songs.csv")
    try:
        with open(summary_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_header)
            writer.writeheader(); writer.writerows(sorted_results_for_csv)
        print(f"Full analysis summary for {processed_count} songs saved to {summary_csv_path}")
    except IOError: print(f"Error: Could not write CSV to {summary_csv_path}")

    actual_top_n = min(TOP_N_SONGS, len(sorted_results_for_csv))
    if actual_top_n > 0:
        top_n_best = sorted_results_for_csv[:actual_top_n]
        best_csv_path = os.path.join(output_root_dir, f"top_{actual_top_n}_best_songs_by_avg_boundary_error.csv") # Renamed
        try:
            with open(best_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_header); writer.writeheader(); writer.writerows(top_n_best)
            print(f"Top {len(top_n_best)} best songs saved to {best_csv_path}")
        except IOError: print(f"Error writing {best_csv_path}")

        top_n_worst = sorted_results_for_csv[-actual_top_n:] 
        top_n_worst.reverse() 
        worst_csv_path = os.path.join(output_root_dir, f"top_{actual_top_n}_worst_songs_by_avg_boundary_error.csv") # Renamed
        try:
            with open(worst_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_header); writer.writeheader(); writer.writerows(top_n_worst)
            print(f"Top {len(top_n_worst)} worst songs (from worst to less-worst) saved to {worst_csv_path}")
        except IOError: print(f"Error writing {worst_csv_path}")
    else:
        print("Not enough songs to generate best/worst lists.")

    if not results_df.empty:
        stats_output_path = os.path.join(output_root_dir, "aggregate_performance_metrics.txt")
        generate_and_save_statistics(results_df, stats_output_path, dataset_name, processed_count)
        
        generate_histograms(results_df, plots_output_dir)
        generate_scatter_plots(results_df, plots_output_dir)
        
        errors_for_cumulative_plot = results_df['error_avg_start_end_s'].astype(float).dropna().tolist()
        if errors_for_cumulative_plot:
             generate_cumulative_error_plot(errors_for_cumulative_plot, plots_output_dir)
        else:
            print("Skipping cumulative boundary error plot as there are no valid data points.")

    else:
        print("Results DataFrame is empty, skipping generation of aggregate stats and plots.")


    if valid_attn_entries > 0:
        avg_attn_in_pred = total_attn_in_pred / valid_attn_entries
        avg_attn_in_gt = total_attn_in_gt / valid_attn_entries
        print("\n--- Attention Overlap Summary ---")
        print(f"Average Attention Score within Predicted Highlight Window: {avg_attn_in_pred:.4f}")
        print(f"Average Attention Score within Ground Truth Highlight Window: {avg_attn_in_gt:.4f}")
        if avg_attn_in_gt > avg_attn_in_pred: print("On average, attention scores overlap better with Ground Truth highlights.")
        elif avg_attn_in_pred > avg_attn_in_gt: print("On average, attention scores overlap better with Predicted highlights.")
        else: print("On average, attention scores overlap similarly with Predicted and Ground Truth highlights.")
    else: print("\nCould not compute attention overlap summary (no valid attention entries).")
    
    # --- NEW: Explicit MAE reporting ---
    if not results_df.empty:
        print("\n--- Overall Performance Summary (Mean Absolute Errors) ---")
        if 'error_avg_start_end_s' in results_df.columns:
            mean_avg_boundary_error = results_df['error_avg_start_end_s'].mean()
            if not np.isnan(mean_avg_boundary_error):
                 print(f"Mean Average Boundary Error (start/end): {mean_avg_boundary_error:.3f} s")
            else:
                 print(f"Mean Average Boundary Error (start/end): N/A (no valid data)")
        if 'error_center_s' in results_df.columns:
            mean_center_error = results_df['error_center_s'].mean()
            if not np.isnan(mean_center_error):
                print(f"Mean Center Time Error:                  {mean_center_error:.3f} s")
            else:
                print(f"Mean Center Time Error:                  N/A (no valid data)")

        if 'error_duration_s' in results_df.columns:
            mean_duration_error = results_df['error_duration_s'].mean()
            if not np.isnan(mean_duration_error):
                print(f"Mean Highlight Duration Error:           {mean_duration_error:.3f} s")
            else:
                print(f"Mean Highlight Duration Error:           N/A (no valid data)")
    # --- END NEW ---

    print(f"\nAnalysis complete. {processed_count} songs from '{dataset_name}' set processed. Outputs in '{output_root_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze MusicHighlighter model performance on test/validation set.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pre-trained .pt model.")
    parser.add_argument("--split_file", type=str, default="train_val_test_split.json", help="Path to split JSON (e.g., train_val_test_split.json). Will prioritize 'test' set, then 'val'.")
    parser.add_argument("--max_songs", type=int, default=None, help="Maximum total songs to process from the dataset for metrics and detailed output. Default is all.")
    parser.add_argument("--output_dir", type=str, default="analysis_output", help="Root output directory for all analysis files.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage.")
    args = parser.parse_args()
    device_name = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    
    print(f"Starting analysis with model: {args.model_path}")
    print(f"Using split file: {args.split_file}")
    print(f"Output directory: {args.output_dir}")
    if args.max_songs is not None:
        print(f"Maximum songs to process: {args.max_songs}")
    print(f"Using device: {device_name}")

    analyze_songs(args.model_path, args.split_file, args.output_dir, device_name, max_songs_to_process=args.max_songs)