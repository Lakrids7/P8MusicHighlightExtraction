#!/usr/bin/env python3
# evaluate_model.py
import os, json, math, argparse, re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Constants (same as training script)
N_MELS, CHUNK_FRAMES, TARGET_DIM, MEL_SUFFIX = 128, 128, 2, ".npy"

# Helpers
def parse_metadata(meta_path):
    """
    Parses metadata to get normalized start/end times and total duration.
    Returns: (np.array([norm_start, norm_end]), total_duration_seconds) or None
    """
    try:
        with open(meta_path) as f: text = f.read()
        start = float(re.search(r"Detected Timestamp:\s*([\d.]+)", text).group(1))
        end = float(re.search(r"Preview End Timestamp:\s*([\d.]+)", text).group(1))
        total_dur = float(re.search(r"Full Song Duration:\s*([\d.]+)", text).group(1))
        if total_dur > 0 and end > start:
            return np.array([start / total_dur, end / total_dur], dtype=np.float32), total_dur
    except Exception: pass
    return None, None

# Dataset
class SongDataset(Dataset):
    def __init__(self, song_dirs):
        self.items = []
        for d in song_dirs:
            mel_path = os.path.join(d, f"full_song{MEL_SUFFIX}")
            meta_path = os.path.join(d, "metadata.txt")
            if os.path.exists(mel_path) and os.path.exists(meta_path):
                self.items.append((mel_path, meta_path))
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        mel_path, meta_path = self.items[idx]
        try:
            mel_chunks = np.load(mel_path)
            label, total_duration_sec = parse_metadata(meta_path) # Modified to get total_duration_sec
            if label is not None and total_duration_sec is not None and \
               mel_chunks.ndim == 3 and mel_chunks.shape[1:] == (N_MELS, CHUNK_FRAMES):
                return (torch.from_numpy(mel_chunks),
                        torch.from_numpy(label),
                        torch.tensor(total_duration_sec, dtype=torch.float32))
        except Exception as e:
            # print(f"Warning: Skipping item {mel_path} due to error: {e}") # Optional: for debugging
            pass
        return None

# Collate
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None, None, None, None
    mels, labels, total_durations = zip(*batch) # Modified to unpack total_durations
    lengths = torch.tensor([mel.shape[0] for mel in mels])
    padded_mels = pad_sequence(mels, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels)
    total_durations = torch.stack(total_durations) # Stack total_durations
    return padded_mels, labels, lengths, total_durations

# Model (copied exactly from your training script)
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
        self.regr_head = nn.Sequential(nn.Linear(self.feat_dim, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, TARGET_DIM), nn.Sigmoid())

    def forward(self, x, lengths): # x: (B, L, M, T)
        B, L_max, M, T = x.shape; dev = x.device
        x = x.permute(0, 1, 3, 2).reshape(B * L_max, 1, T, M) # (B*L, 1, T, M)
        x = torch.max(self.conv_blocks(x).squeeze(3), dim=2)[0] # (B*L, feat_dim)
        h_t = x.view(B, L_max, -1) # (B, L, feat_dim)
        pe = positional_encoding(L_max, self.feat_dim, dev).unsqueeze(0) # (1, L, D)
        attn_logits = self.attn_mlp((h_t + pe).view(B * L_max, -1)).view(B, L_max) # (B, L)
        mask = torch.arange(L_max, device=dev)[None, :] >= lengths[:, None] # (B, L)
        attn_logits.masked_fill_(mask, -float('inf'))
        alpha_t = torch.softmax(attn_logits, dim=1).unsqueeze(1) # (B, 1, L)
        weighted_features = torch.bmm(alpha_t, h_t).squeeze(1) # (B, D)
        return self.regr_head(weighted_features), alpha_t.squeeze(1)

# Evaluation Function
def evaluate_mae_seconds(model, dataloader, device, dataset_name="Dataset"):
    model.eval()
    total_mae_seconds = 0.0
    total_samples = 0 # This will count individual start/end time predictions

    with torch.no_grad():
        for mel, y_norm, lengths, total_durations_sec in dataloader:
            if mel is None: continue # Should not happen if collate_fn filters empty batches

            mel, y_norm, lengths, total_durations_sec = \
                mel.to(device), y_norm.to(device), lengths.to(device), total_durations_sec.to(device)

            pred_norm, _ = model(mel, lengths) # pred_norm is [B, TARGET_DIM]

            # Convert predictions and labels to seconds
            # total_durations_sec is [B], need to make it [B, 1] to broadcast with [B, TARGET_DIM]
            pred_seconds = pred_norm * total_durations_sec.unsqueeze(1)
            y_seconds = y_norm * total_durations_sec.unsqueeze(1)

            # Calculate MAE for this batch in seconds
            # abs error for each start/end time, then sum them up
            mae_batch_seconds = torch.abs(pred_seconds - y_seconds).sum().item()
            total_mae_seconds += mae_batch_seconds
            total_samples += y_norm.numel() # B * TARGET_DIM (e.g., batch_size * 2)

    if total_samples == 0:
        print(f"No samples processed for {dataset_name}.")
        return float('inf')

    avg_mae_seconds = total_mae_seconds / total_samples
    print(f"{dataset_name} - Average MAE: {avg_mae_seconds:.4f} seconds (over {total_samples // TARGET_DIM} songs, {total_samples} values)")
    return avg_mae_seconds


# Main execution
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate MusicHighlighter (precomputed, short)")
    p.add_argument("--model_path", default="highlighter_regr_precomp_short.pt", help="Path to the trained model .pt file")
    p.add_argument("--split", default="train_val_test_split.json", help="Path to split JSON")
    p.add_argument("--batch", type=int, default=8, help="Batch size for evaluation")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 for main process)") # Often 0 is better for eval
    p.add_argument("--no_val", action="store_true", help="Skip validation set evaluation")
    p.add_argument("--no_test", action="store_true", help="Skip test set evaluation")
    args = p.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"Using device: {device}")

    # Load model
    model = MusicHighlighter().to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    # Load split file
    try:
        split_data = json.load(open(args.split))
    except Exception as e:
        print(f"Error loading split file {args.split}: {e}")
        exit(1)

    ds_args = {
        'batch_size': args.batch,
        'num_workers': args.workers,
        'pin_memory': True if device == 'cuda' else False, # Pin memory only if using CUDA
        'collate_fn': collate_fn,
        'persistent_workers': args.workers > 0
    }

    # Evaluate on Validation Set
    if not args.no_val:
        val_song_dirs = split_data.get("val", [])
        if val_song_dirs:
            val_ds = SongDataset(val_song_dirs)
            if len(val_ds) > 0:
                print(f"Found {len(val_ds)} validation samples.")
                val_ld = DataLoader(val_ds, shuffle=False, **ds_args)
                evaluate_mae_seconds(model, val_ld, device, "Validation Set")
            else:
                print("No valid validation samples found to evaluate.")
        else:
            print("No 'val' key found in split file or validation set is empty.")

    # Evaluate on Test Set
    if not args.no_test:
        test_song_dirs = split_data.get("test", [])
        if test_song_dirs:
            test_ds = SongDataset(test_song_dirs)
            if len(test_ds) > 0:
                print(f"Found {len(test_ds)} test samples.")
                test_ld = DataLoader(test_ds, shuffle=False, **ds_args)
                evaluate_mae_seconds(model, test_ld, device, "Test Set")
            else:
                print("No valid test samples found to evaluate.")
        else:
            print("No 'test' key found in split file or test set is empty.")

    print("Evaluation complete.")