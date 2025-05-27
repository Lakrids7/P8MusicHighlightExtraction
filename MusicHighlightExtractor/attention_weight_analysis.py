import os
import json
import math
import argparse
import re
import glob
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

# --- Constants ---
N_MELS = 128
CHUNK_FRAMES = 128
TARGET_DIM = 2 # Still needed for model definition, though not primary focus
MEL_SUFFIX = ".npy"
AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
# SR_TARGET_FOR_DTW = 22050 # DTW removed for simplicity based on request
# N_FFT_DTW = 2048
# HOP_LENGTH_DTW = 512
# N_MELS_DTW = 64
TOP_N_SONGS = 100 # For best/worst lists
ATTENTION_HIGHLIGHT_DURATION_S = 30.0


# --- Model Definition (Same as before, as attention scores are needed) ---
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
        prediction = self.regression_head(weighted_features) # Regression output still computed
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

# --- Function to calculate sum of attention within a time window ---
def calculate_attention_sum_in_window(attention_scores, chunk_center_times, window_start_s, window_end_s):
    if attention_scores is None or chunk_center_times is None or len(attention_scores) != len(chunk_center_times):
        return np.nan
    relevant_attention_sum = 0.0
    for i, center_time in enumerate(chunk_center_times):
        # Check if the chunk's center falls within the window
        if window_start_s <= center_time < window_end_s:
            relevant_attention_sum += attention_scores[i]
    return relevant_attention_sum

# --- Function to find the best N-second attention window ---
def find_best_attention_window(attention_scores, chunk_start_times, actual_chunk_duration_s, song_duration_s, target_window_duration_s=30.0):
    num_chunks = len(attention_scores)

    if num_chunks == 0 or actual_chunk_duration_s <= 0:
        end_s = min(target_window_duration_s, song_duration_s if song_duration_s > 0 else target_window_duration_s)
        return 0.0, end_s, end_s / 2.0, 0.0 # start_s, end_s, center_s, sum_attn

    num_chunks_in_target_window = max(1, int(round(target_window_duration_s / actual_chunk_duration_s)))
    effective_chunks_in_window = min(num_chunks_in_target_window, num_chunks)

    best_sum_attn = -float('inf')
    best_start_idx = 0

    if num_chunks <= effective_chunks_in_window: # Song shorter than or equal to target window
        best_sum_attn = np.sum(attention_scores)
        best_start_idx = 0
        effective_chunks_in_window = num_chunks # The window is all chunks
    else:
        for i in range(num_chunks - effective_chunks_in_window + 1):
            current_sum = np.sum(attention_scores[i : i + effective_chunks_in_window])
            if current_sum > best_sum_attn:
                best_sum_attn = current_sum
                best_start_idx = i
    
    attn_highlight_start_s = chunk_start_times[best_start_idx]
    attn_highlight_end_s = attn_highlight_start_s + (effective_chunks_in_window * actual_chunk_duration_s)
    attn_highlight_end_s = min(attn_highlight_end_s, song_duration_s)

    if attn_highlight_start_s >= attn_highlight_end_s : # Handle potential issues if song_duration_s is tiny
        if song_duration_s > 0 :
             attn_highlight_start_s = max(0.0, attn_highlight_end_s - (effective_chunks_in_window * actual_chunk_duration_s))
             if attn_highlight_start_s >= attn_highlight_end_s:
                attn_highlight_start_s = 0.0
                attn_highlight_end_s = min(0.01 * effective_chunks_in_window , song_duration_s) # small duration
        else: # song_duration_s is 0 or less
            attn_highlight_start_s = 0.0
            attn_highlight_end_s = 0.0
            
    attn_highlight_center_s = (attn_highlight_start_s + attn_highlight_end_s) / 2.0
    
    if best_sum_attn == -float('inf'): best_sum_attn = 0.0 # Ensure it's a number

    return attn_highlight_start_s, attn_highlight_end_s, attn_highlight_center_s, best_sum_attn


# --- Main Analysis Function ---
def analyze_songs(model_path, split_file_path, output_root_dir, device_str, max_songs_to_process=None):
    device = torch.device(device_str)
    model = MusicHighlighter().to(device); model.eval()
    try: model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e: print(f"Error loading model: {e}"); return

    try:
        with open(split_file_path, 'r') as f: split_data = json.load(f)
        val_song_dirs = split_data.get("val", [])
        if not val_song_dirs: print("No validation songs in split."); return
    except Exception as e: print(f"Error loading split: {e}"); return

    dataset = AnalysisSongDataset(val_song_dirs)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=analysis_collate_fn)
    if len(dataset) == 0: print("No valid songs found in the dataset."); return
    
    os.makedirs(output_root_dir, exist_ok=True)
    
    all_song_results_data = []
    csv_header = [
        "song_id", "song_duration_s",
        "gt_start_s", "gt_end_s", "gt_center_s",
        "attn_highlight_start_s", "attn_highlight_end_s", "attn_highlight_center_s",
        "error_attn_center_vs_gt_center_s",
        "sum_attn_in_gt_window", "sum_attn_in_attn_highlight_window"
    ]

    num_to_process = len(dataset)
    if max_songs_to_process is not None and max_songs_to_process < len(dataset):
        num_to_process = max_songs_to_process
        print(f"Analyzing the first {num_to_process} of {len(dataset)} valid songs...")
    else:
        print(f"Analyzing all {len(dataset)} valid songs...")

    processed_count = 0
    total_sum_attn_in_gt = 0; total_sum_attn_in_attn_highlight = 0; valid_attn_sum_entries = 0

    for i, data_item in enumerate(dataloader):
        if max_songs_to_process is not None and processed_count >= max_songs_to_process:
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
            _, attn_scores_tensor = model(mel_for_model, num_chunks_tensor) # We mainly need attention scores

        gt_norm_np = gt_highlight_norm_tensor.cpu().numpy()
        attn_scores_np = attn_scores_tensor.squeeze(0)[:num_chunks_val].cpu().numpy()

        gt_start_s = gt_norm_np[0] * song_duration_s
        gt_end_s = gt_norm_np[1] * song_duration_s
        gt_center_s = (gt_start_s + gt_end_s) / 2.0

        chunk_start_times = None; chunk_center_times = None; actual_chunk_duration_s = 0
        attn_highlight_start_s, attn_highlight_end_s, attn_highlight_center_s = np.nan, np.nan, np.nan
        sum_attn_in_attn_highlight_window = np.nan
        sum_attn_in_gt_window = np.nan
        error_attn_center_vs_gt_center_s = np.nan

        if num_chunks_val > 0:
            actual_chunk_duration_s = song_duration_s / num_chunks_val
            chunk_start_times = np.arange(num_chunks_val) * actual_chunk_duration_s
            chunk_center_times = chunk_start_times + actual_chunk_duration_s / 2.0
            
            attn_highlight_start_s, attn_highlight_end_s, attn_highlight_center_s, sum_attn_in_attn_highlight_window = \
                find_best_attention_window(attn_scores_np, chunk_start_times, actual_chunk_duration_s, song_duration_s, ATTENTION_HIGHLIGHT_DURATION_S)
            
            sum_attn_in_gt_window = calculate_attention_sum_in_window(attn_scores_np, chunk_center_times, gt_start_s, gt_end_s)
            
            if not np.isnan(attn_highlight_center_s) and not np.isnan(gt_center_s):
                 error_attn_center_vs_gt_center_s = abs(attn_highlight_center_s - gt_center_s)

            if not np.isnan(sum_attn_in_gt_window) and not np.isnan(sum_attn_in_attn_highlight_window):
                total_sum_attn_in_gt += sum_attn_in_gt_window
                total_sum_attn_in_attn_highlight += sum_attn_in_attn_highlight_window
                valid_attn_sum_entries +=1
        
        song_out_dir = os.path.join(output_root_dir, song_id)
        os.makedirs(song_out_dir, exist_ok=True)
        
        try:
            y_audio, sr_orig = librosa.load(audio_file_path, sr=None, mono=True)
            def get_segment(y, sr, start_s, end_s):
                start_sample = max(0, int(start_s * sr)); end_sample = min(len(y), int(end_s * sr))
                return y[start_sample:end_sample] if end_sample > start_sample else np.array([])
            
            gt_audio_segment = get_segment(y_audio, sr_orig, gt_start_s, gt_end_s)
            if gt_audio_segment.size > 0:
                 sf.write(os.path.join(song_out_dir, "highlight_ground_truth.wav"), gt_audio_segment, sr_orig)

            # Optionally, save the attention-derived highlight audio
            # attn_highlight_audio_segment = get_segment(y_audio, sr_orig, attn_highlight_start_s, attn_highlight_end_s)
            # if attn_highlight_audio_segment.size > 0:
            #     sf.write(os.path.join(song_out_dir, "highlight_attention_derived.wav"), attn_highlight_audio_segment, sr_orig)

        except Exception as e: print(f"Audio processing error for {song_id}: {e}")

        try:
            time_ax_attn = chunk_center_times if chunk_center_times is not None else \
                           (np.arange(len(attn_scores_np)) * actual_chunk_duration_s + actual_chunk_duration_s/2.0 if num_chunks_val > 0 and actual_chunk_duration_s > 0 else np.array([]))
            
            if len(time_ax_attn) == len(attn_scores_np) and len(attn_scores_np) > 0:
                plt.figure(figsize=(12, 6))
                plt.plot(time_ax_attn, attn_scores_np, label='Attention Score', color='dodgerblue', lw=2)
                plt.axvspan(gt_start_s, gt_end_s, color='mediumseagreen', alpha=0.4, label=f'GT ({gt_start_s:.1f}-{gt_end_s:.1f}s)')
                if not np.isnan(attn_highlight_start_s) and not np.isnan(attn_highlight_end_s):
                    plt.axvspan(attn_highlight_start_s, attn_highlight_end_s, color='salmon', alpha=0.4, 
                                label=f'Attention Highlight ({attn_highlight_start_s:.1f}-{attn_highlight_end_s:.1f}s)')
                
                plt.xlabel("Time (s)"); plt.ylabel("Attention Score"); plt.title(f"Attention Scores vs Highlights for {song_id}")
                plt.legend(); plt.grid(True, linestyle=':', alpha=0.7); plt.tight_layout()
                plt.savefig(os.path.join(song_out_dir, "attention_scores_vs_highlights.png")); plt.close()
            elif len(attn_scores_np) > 0: print(f"Warning: Could not plot attention for {song_id} due to time axis mismatch or empty data.")
        except Exception as e: print(f"Plotting error for {song_id}: {e}")

        with open(os.path.join(song_out_dir, "attention_highlight_info.txt"), 'w') as f:
            f.write(f"Song ID: {song_id}\nDuration: {song_duration_s:.2f}s\n\n")
            f.write(f"Ground Truth Highlight:\n"
                    f"  Start: {gt_start_s:.2f}s, End: {gt_end_s:.2f}s, Center: {gt_center_s:.2f}s\n\n")
            f.write(f"Attention-Derived {ATTENTION_HIGHLIGHT_DURATION_S:.0f}s Highlight:\n"
                    f"  Start: {attn_highlight_start_s:.2f}s, End: {attn_highlight_end_s:.2f}s, Center: {attn_highlight_center_s:.2f}s\n\n")
            f.write(f"Comparison:\n"
                    f"  Error (Attention Highlight Center vs GT Center): {error_attn_center_vs_gt_center_s:.2f}s\n\n")
            f.write(f"Attention Sums:\n"
                    f"  In GT Window: {sum_attn_in_gt_window:.4f}\n"
                    f"  In Attention Highlight Window: {sum_attn_in_attn_highlight_window:.4f}\n")

        all_song_results_data.append({
            "song_id": song_id, "song_duration_s": song_duration_s,
            "gt_start_s": gt_start_s, "gt_end_s": gt_end_s, "gt_center_s": gt_center_s,
            "attn_highlight_start_s": attn_highlight_start_s, "attn_highlight_end_s": attn_highlight_end_s, 
            "attn_highlight_center_s": attn_highlight_center_s,
            "error_attn_center_vs_gt_center_s": error_attn_center_vs_gt_center_s,
            "sum_attn_in_gt_window": sum_attn_in_gt_window,
            "sum_attn_in_attn_highlight_window": sum_attn_in_attn_highlight_window
        })
        processed_count += 1
        print(f"Processed ({processed_count}/{num_to_process}): {song_id} | Err Attn Ctr: {error_attn_center_vs_gt_center_s:.2f}s | Attn Sum GT: {sum_attn_in_gt_window:.2f} | Attn Sum AttnHigh: {sum_attn_in_attn_highlight_window:.2f}")

    if not all_song_results_data:
        print("No songs were processed successfully. Exiting.")
        return

    # Sort by the new error metric
    sorted_results = sorted(all_song_results_data, key=lambda x: float('inf') if np.isnan(x["error_attn_center_vs_gt_center_s"]) else x["error_attn_center_vs_gt_center_s"])

    summary_csv_path = os.path.join(output_root_dir, "analysis_summary_all_processed_songs.csv")
    try:
        with open(summary_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_header)
            writer.writeheader(); writer.writerows(sorted_results)
        print(f"Full analysis summary for {processed_count} songs saved to {summary_csv_path}")
    except IOError: print(f"Error: Could not write CSV to {summary_csv_path}")

    actual_top_n = min(TOP_N_SONGS, len(sorted_results))
    if actual_top_n > 0:
        top_n_best = sorted_results[:actual_top_n]
        best_csv_path = os.path.join(output_root_dir, f"top_{actual_top_n}_best_songs_by_attn_center_error.csv")
        try:
            with open(best_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_header); writer.writeheader(); writer.writerows(top_n_best)
            print(f"Top {len(top_n_best)} best songs saved to {best_csv_path}")
        except IOError: print(f"Error writing {best_csv_path}")

        top_n_worst = sorted_results[-actual_top_n:] 
        top_n_worst.reverse() # To show worst first
        worst_csv_path = os.path.join(output_root_dir, f"top_{actual_top_n}_worst_songs_by_attn_center_error.csv")
        try:
            with open(worst_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_header); writer.writeheader(); writer.writerows(top_n_worst)
            print(f"Top {len(top_n_worst)} worst songs saved to {worst_csv_path}")
        except IOError: print(f"Error writing {worst_csv_path}")

    if valid_attn_sum_entries > 0:
        avg_sum_attn_in_gt = total_sum_attn_in_gt / valid_attn_sum_entries
        avg_sum_attn_in_attn_highlight = total_sum_attn_in_attn_highlight / valid_attn_sum_entries
        print("\n--- Attention Sum Summary ---")
        print(f"Average Sum of Attention Scores within Ground Truth Highlight Window: {avg_sum_attn_in_gt:.4f}")
        print(f"Average Sum of Attention Scores within {ATTENTION_HIGHLIGHT_DURATION_S:.0f}s Attention-Derived Highlight Window: {avg_sum_attn_in_attn_highlight:.4f}")
        if avg_sum_attn_in_gt > avg_sum_attn_in_attn_highlight: 
            print("On average, GT windows capture slightly more total attention than the best fixed-duration attention windows.")
        elif avg_sum_attn_in_attn_highlight > avg_sum_attn_in_gt:
             print("On average, the best fixed-duration attention windows capture more total attention than GT windows.")
        else:
             print("On average, attention sums are similar between GT and attention-derived windows.")
    else: print("\nCould not compute attention sum summary (no valid attention sum entries).")
    
    print(f"\nAnalysis complete. {processed_count} songs processed with detailed outputs in subfolders.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze MusicHighlighter model attention performance.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pre-trained .pt model.")
    parser.add_argument("--split_file", type=str, default="train_val_test_split.json", help="Path to split JSON.")
    parser.add_argument("--max_songs", type=int, default=None, help="Maximum total songs to process from the validation set. Default is all.")
    parser.add_argument("--output_dir", type=str, default="analysis_output_attention", help="Output directory for attention analysis.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    args = parser.parse_args()
    device_name = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    analyze_songs(args.model_path, args.split_file, args.output_dir, device_name, max_songs_to_process=args.max_songs)