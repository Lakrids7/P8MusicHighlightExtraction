#!/usr/bin/env python3
# train_precomputed_short.py
import os, json, math, argparse, re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Constants
N_MELS, CHUNK_FRAMES, TARGET_DIM, MEL_SUFFIX = 128, 128, 2, ".npy"

# Helpers
def parse_metadata(meta_path):
    try:
        with open(meta_path) as f: text = f.read()
        start = float(re.search(r"Detected Timestamp:\s*([\d.]+)", text).group(1))
        end = float(re.search(r"Preview End Timestamp:\s*([\d.]+)", text).group(1))
        total_dur = float(re.search(r"Full Song Duration:\s*([\d.]+)", text).group(1))
        if total_dur > 0 and end > start:
            return np.array([start / total_dur, end / total_dur], dtype=np.float32)
    except Exception: pass
    return None

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
            label = parse_metadata(meta_path)
            if label is not None and mel_chunks.ndim == 3 and mel_chunks.shape[1:] == (N_MELS, CHUNK_FRAMES):
                return torch.from_numpy(mel_chunks), torch.from_numpy(label)
        except Exception: pass
        return None

# Collate
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None, None, None
    mels, labels = zip(*batch)
    lengths = torch.tensor([mel.shape[0] for mel in mels])
    padded_mels = pad_sequence(mels, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels)
    return padded_mels, labels, lengths

# Model
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

# Training Loop
def run_training(split_path, epochs, batch_size, lr, device, workers, out):
    try: split = json.load(open(split_path))
    except Exception as e: return print(f"Error loading split {split_path}: {e}")
    ds_args = {'batch_size': batch_size, 'num_workers': workers, 'pin_memory': True, 'collate_fn': collate_fn, 'persistent_workers': workers > 0}
    train_ds = SongDataset(split.get("train", [])); val_ds = SongDataset(split.get("val", []))
    if not train_ds: return print("No valid training samples found.")
    print(f"Using {len(train_ds)} train, {len(val_ds)} val samples.")
    train_ld = DataLoader(train_ds, shuffle=True, **ds_args); val_ld = DataLoader(val_ds, shuffle=False, **ds_args)
    model = MusicHighlighter().to(device); opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss(); best_val_loss = float("inf")
    print(f"Starting training on {device}...")
    for ep in range(1, epochs + 1):
        model.train(); train_loss, train_n = 0.0, 0
        for mel, y, lengths in train_ld:
            if mel is None: continue
            mel, y, lengths = mel.to(device), y.to(device), lengths.to(device)
            opt.zero_grad(); pred, _ = model(mel, lengths); loss = crit(pred, y)
            loss.backward(); opt.step()
            train_loss += loss.item() * mel.size(0); train_n += mel.size(0)
        avg_train_loss = train_loss / train_n if train_n else 0

        model.eval(); val_loss, val_n = 0.0, 0
        with torch.no_grad():
            for mel, y, lengths in val_ld:
                if mel is None: continue
                mel, y, lengths = mel.to(device), y.to(device), lengths.to(device)
                pred, _ = model(mel, lengths); loss = crit(pred, y)
                val_loss += loss.item() * mel.size(0); val_n += mel.size(0)
        avg_val_loss = val_loss / val_n if val_n else float('inf')
        print(f"E {ep:02d}/{epochs} | Tr Loss: {avg_train_loss:.4f} | Vl Loss: {avg_val_loss:.4f}", end="")
        if avg_val_loss < best_val_loss and val_n > 0:
            best_val_loss = avg_val_loss; torch.save(model.state_dict(), out); print(f" -> Saved {out}")
        else: print()
    print(f"Done. Best Val Loss: {best_val_loss:.4f}")

# CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train MusicHighlighter (precomputed, short)")
    p.add_argument("--split", default="train_val_test_split.json", help="Path to split JSON")
    p.add_argument("--epochs", type=int, default=200, help="Epochs")
    p.add_argument("--batch", type=int, default=8, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--out", default="highlighter_regr", help="Output model path")
    args = p.parse_args()
    dev = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    run_training(args.split, args.epochs, args.batch, args.lr, dev, args.workers, args.out)