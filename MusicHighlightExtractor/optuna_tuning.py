#!/usr/bin/env python3
# music_highlighter_torch_regression_full_optuna_short_multi_gpu.py
import os, json, math, argparse, re, pathlib, warnings
import numpy as np
import librosa, torch, optuna
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from optuna.exceptions import TrialPruned
# Added for AMP
from torch.cuda.amp import GradScaler, autocast

warnings.filterwarnings("ignore", category=UserWarning)

# Constants
SR, HOP, N_MELS, CHUNK_F = 22_050, 512, 128, 128
TARGET_DIM = 2

# Helpers (Identical to previous)
def audio_to_chunks(p: str) -> np.ndarray|None:
    try:
        y, _ = librosa.load(p, sr=SR, mono=True); mel = librosa.feature.melspectrogram(y=y, sr=SR, hop_length=HOP, n_mels=N_MELS).T
        n_f = mel.shape[0]; mel = librosa.power_to_db(mel[:n_f // CHUNK_F * CHUNK_F], ref=np.max) if n_f >= CHUNK_F else None
        return mel.reshape(-1, CHUNK_F, N_MELS).transpose(0, 2, 1).astype(np.float32) if mel is not None else None
    except Exception as e: print(f"Err audio {p}: {e}"); return None
def parse_meta(p: str) -> tuple[np.ndarray, float]|None:
    try:
        txt = pathlib.Path(p).read_text()
        s = float(re.search(r"Detected Timestamp:\s*([\d.]+)", txt).group(1))
        e = float(re.search(r"Preview End Timestamp:\s*([\d.]+)", txt).group(1))
        d = float(re.search(r"Full Song Duration:\s*([\d.]+)", txt).group(1))
        return (np.array([s/d, e/d], dtype=np.float32), d) if d > 0 and 0 <= s < e <= d else None
    except Exception: return None # Suppress parse errors for brevity here

# Dataset & Collate (Dataset slightly simplified init, Collate identical)
class SongDataset(Dataset):
    def __init__(self, dirs):
        self.items = []
        for d in dirs:
            meta_p = os.path.join(d, "metadata.txt")
            audio_p = next((os.path.join(d, f) for f in ["full_song.wav", "full_song.mp3"] if os.path.exists(os.path.join(d, f))), None)
            if audio_p and os.path.exists(meta_p) and parse_meta(meta_p) is not None: self.items.append((audio_p, meta_p))
        # print(f"Dataset init: Found {len(self.items)} valid items from {len(dirs)} dirs.") # Optional debug
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        ap, mp = self.items[i]; mel, meta = audio_to_chunks(ap), parse_meta(mp)
        return (torch.from_numpy(mel), torch.from_numpy(meta[0]), meta[1]) if mel is not None and meta is not None else None
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None, None, None, None
    mels, lbls_norm, durs = zip(*batch)
    lens = torch.tensor([m.shape[0] for m in mels], dtype=torch.long)
    mels_pad = pad_sequence([torch.as_tensor(m) for m in mels], batch_first=True)
    return mels_pad, torch.stack([torch.as_tensor(l) for l in lbls_norm]), torch.tensor(durs, dtype=torch.float32), lens

# Model (Identical to previous)
def pos_enc(L, D, dev):
    pos = torch.arange(L, device=dev).unsqueeze(1); div = torch.exp(torch.arange(0, D, 2, device=dev) * (-math.log(10000.0) / D))
    pe = torch.zeros(L, D, device=dev); pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div); return pe
class MusicHighlighter(nn.Module):
    def __init__(self, dim=64, dropout=0.5):
        super().__init__(); D = dim * 4
        def CB(i, o, k, s): return nn.Sequential(nn.Conv2d(i, o, k, s, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))
        self.conv = nn.Sequential(CB(1, dim, 3, 2), CB(dim, dim*2, 3, 2), CB(dim*2, D, 3, 2), nn.AdaptiveMaxPool2d(1))
        self.attn = nn.Sequential(nn.Linear(D, D), nn.ReLU(True), nn.Dropout(dropout), nn.Linear(D, D), nn.Tanh(), nn.Dropout(dropout), nn.Linear(D, 1))
        self.regr = nn.Sequential(nn.Linear(D, 128), nn.ReLU(True), nn.Dropout(dropout), nn.Linear(128, TARGET_DIM), nn.Sigmoid())
        self.D = D
    def forward(self, x, lens): # x: (B, Lmax, N_MELS, CHUNK_F)
        B, Lmax, H, W = x.shape; dev = x.device; h = self.conv(x.view(B * Lmax, 1, H, W)).view(B, Lmax, self.D)
        logits = self.attn((h + pos_enc(Lmax, self.D, dev)).view(B * Lmax, self.D)).view(B, Lmax)
        mask = torch.arange(Lmax, device=dev)[None, :] >= lens[:, None]; logits.masked_fill_(mask, -float('inf'))
        alpha = torch.softmax(logits, dim=1); ctx = torch.bmm(alpha.unsqueeze(1), h).squeeze(1); return self.regr(ctx), alpha

# Optuna Objective (Modified for DataParallel and AMP)
def objective(trial, args, split, dev, n_gpu): # Added n_gpu
    hp = {'lr': trial.suggest_float("lr", 1e-5, 1e-3, log=True),
          # Adjust batch size suggestion if OOM persists: try smaller numbers like [2, 4, 8, 16]
          'bs': trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64]), # Total batch size
          'dim': trial.suggest_categorical("model_dim", [32, 64, 128]),
          'drop': trial.suggest_float("dropout", 0.1, 0.6)}
    print(f"--- Trial {trial.number} Params: {hp} ---")
    train_ds, val_ds = SongDataset(split["train"]), SongDataset(split["val"])
    if len(train_ds) == 0: print("Error: No valid train data!"); return float('inf')

    # Ensure effective batch size per GPU is at least 1
    eff_bs_per_gpu = hp['bs'] // n_gpu if n_gpu > 0 else hp['bs']
    if eff_bs_per_gpu < 1 and n_gpu > 0 :
        print(f"Warning: Skipping trial {trial.number}. Batch size {hp['bs']} < num GPUs {n_gpu}.")
        # Or adjust hp['bs'] = n_gpu if you prefer to always run
        return float('inf') # Skip trial if batch size too small for DP

    train_ld = DataLoader(train_ds, hp['bs'], shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
    val_ld = DataLoader(val_ds, hp['bs'], num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)

    model_base = MusicHighlighter(dim=hp['dim'], dropout=hp['drop'])

    # --- DataParallel Wrapping ---
    if n_gpu > 1:
        print(f"Using {n_gpu} GPUs via DataParallel.")
        model = nn.DataParallel(model_base) # Wrap before moving
    else:
        model = model_base
    model.to(dev) # Move wrapped or base model to primary device
    # ---------------------------

    opt = torch.optim.Adam(model.parameters(), lr=hp['lr']); crit = nn.MSELoss(); best_loss = float("inf")
    scaler = GradScaler(enabled=(dev == 'cuda')) # AMP Scaler

    for ep in range(args.epochs):
        model.train(); train_loss, train_n = 0.0, 0
        for m, y, _, lens in train_ld:
            if m is None: continue
            # Move data to the *primary* device (DP handles scattering)
            m, y, lens = m.to(dev), y.to(dev), lens.to(dev)
            opt.zero_grad()
            # --- AMP Forward ---
            with autocast(enabled=(dev == 'cuda')):
                pred, _ = model(m, lens)
                loss = crit(pred, y)
            # --- AMP Backward ---
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            # ------------------
            train_loss += loss.item() * m.size(0); train_n += m.size(0)
        if train_n == 0 and ep == 0 : print("Warning: train_n is 0 after first epoch!"); break

        model.eval(); val_loss, val_mae_sec, val_n = 0.0, 0.0, 0
        with torch.no_grad(), autocast(enabled=(dev == 'cuda')): # AMP for validation too
            for m, y, durs, lens in val_ld:
                if m is None: continue
                m, y, durs, lens = m.to(dev), y.to(dev), durs.to(dev), lens.to(dev)
                pred, _ = model(m, lens); loss = crit(pred, y)
                mae = torch.abs(pred * durs.unsqueeze(1) - y * durs.unsqueeze(1)).sum().item()
                val_loss += loss.item() * m.size(0); val_mae_sec += mae; val_n += m.size(0)

        avg_train = train_loss / train_n if train_n else 0
        avg_val = val_loss / val_n if val_n else float('inf')
        avg_mae = val_mae_sec / (val_n * TARGET_DIM) if val_n else float('inf')
        print(f" E{ep+1:02d}|Tr L:{avg_train:.4f}|Vl L:{avg_val:.4f}|Vl MAE:{avg_mae:.3f}s", end="")

        if val_n > 0:
            is_best = avg_val < best_loss
            if is_best: best_loss = avg_val
            print(" -> Best" if is_best else "")
            trial.report(avg_val, ep)
            if trial.should_prune(): print(" PRUNED"); raise TrialPruned()
        elif len(val_ds) == 0: print(" (No val data)")
        else: print(" (Val batches None)")

    final_mae = avg_mae if val_n > 0 else float('inf')
    if val_n > 0: trial.set_user_attr("val_mae_sec", final_mae)
    final_loss = best_loss if val_n > 0 or len(val_ds)==0 else float('inf')
    print(f"--- Trial {trial.number} Finished. Final Loss Metric: {final_loss:.4f}, MAE: {final_mae:.3f}s ---")
    return final_loss

# Main Execution (Modified for GPU count detection)
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Short MusicHighlighter Tuner (MultiGPU+AMP)")
    p.add_argument("-s", "--split", default="train_val_test_split.json", help="Train/val split JSON")
    p.add_argument("-e", "--epochs", type=int, default=20, help="Epochs per trial")
    p.add_argument("-t", "--n_trials", type=int, default=50, help="Optuna trials")
    p.add_argument("-w", "--workers", type=int, default=os.cpu_count()//2 or 1, help="Dataloader workers")
    p.add_argument("-n", "--study_name", default="mh_tune", help="Optuna study name")
    p.add_argument("--storage", default=None, help="Optuna DB URL (e.g., sqlite:///tune.db)")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    args = p.parse_args()

    # --- Device Setup ---
    n_gpu = 0
    if not args.cpu and torch.cuda.is_available():
        dev = "cuda"
        n_gpu = torch.cuda.device_count()
        print(f"Detected {n_gpu} CUDA devices.")
    else:
        dev = "cpu"
        print("Using CPU.")
    # ---------------------

    try:
        with open(args.split) as f: split_data = json.load(f)
        if "train" not in split_data or "val" not in split_data: raise ValueError("Split file needs 'train' and 'val' keys")
    except Exception as e: exit(f"Error loading split file {args.split}: {e}")

    study = optuna.create_study(study_name=args.study_name, storage=args.storage, direction="minimize", load_if_exists=True, pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    try:
        # Pass n_gpu to objective
        study.optimize(lambda t: objective(t, args, split_data, dev, n_gpu), n_trials=args.n_trials)
    except KeyboardInterrupt: print("\nStudy interrupted by user.")
    except Exception as e: print(f"\nOptuna study error: {e}") ; import traceback; traceback.print_exc() # Print traceback

    print("\n--- Study Summary ---")
    if study.trials:
        valid_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        if valid_trials:
             best = study.best_trial
             mae = best.user_attrs.get('val_mae_sec', 'N/A')
             mae_str = f"{mae:.3f}s" if isinstance(mae, float) else mae
             print(f"Best Trial #{best.number}: Loss={best.value:.6f}, MAE={mae_str}")
             print(f" Params: {best.params}")
        else: print("No trials completed successfully.")
    else: print("No trials were attempted or completed.")
    if args.storage: print(f"Storage: {args.storage}")