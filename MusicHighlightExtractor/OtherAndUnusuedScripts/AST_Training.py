import torch
import torch.nn as nn
import numpy as np
import librosa
import os
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
from datetime import datetime
import time
import traceback # Keep for detailed error logging

# --- NEW: Import transformers ---
from transformers import ASTModel, ASTConfig

# ==============================================================================
#                            CONFIGURATION
# ==============================================================================
CONFIG = {
    "dataset_folder": 'Filtered_Songs2',
    "epochs": 100,             # Number of training epochs
    "batch_size": 16,          # Batch size (adjust based on GPU memory)
    "lr": 5e-5,               # Learning rate
    "weight_decay": 1e-4,     # Weight decay for AdamW
    "ast_model_name": "MIT/ast-finetuned-audioset-10-10-0.4593", # HF model name
    "max_frames": 1024,       # Fixed number of frames for spectrogram input (~23.8s)
    "dropout": 0.3,           # Dropout rate for the regression head
    "n_mels": 128,            # Mel bins
    "sample_rate": 22050,     # Audio sample rate
    "hop_length_mel": 512,    # Hop length for mel spectrogram
    "num_workers": 4,         # DataLoader workers (set to 0 if debugging causes issues)
    "pin_memory": True,       # DataLoader pin_memory
    "seed": 42,               # Random seed for reproducibility
    "run_name_prefix": "ast_regressor", # Prefix for save folder
    "cache_dir": "mel_cache_ast", # Cache directory
    "log_dir": "training_logs_ast" # Base directory for saving results
}
# ==============================================================================

# --- Set Seed for Reproducibility ---
random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])

# --- Data Transformation ---
class NormalizeSpectrogram(nn.Module):
    def forward(self, mel_db):
        mean = mel_db.mean()
        std = mel_db.std()
        return (mel_db - mean) / (std + 1e-9) if std > 1e-9 else mel_db - mean

# --- Dataset ---
class FullSpectrogramDataset(Dataset):
    # Takes the list of song IDs (relative paths) to include
    def __init__(self, root, song_ids_to_include, cache_dir, transform, max_frames, n_mels, sample_rate, hop_length):
        self.root = root
        self.cache_dir = cache_dir
        self.transform = transform
        self.max_frames = max_frames
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        os.makedirs(cache_dir, exist_ok=True)

        # --- CORRECT INITIALIZATION ---
        self.song_list = [] # Initialize the attribute correctly
        for item in song_ids_to_include: # Iterate through the provided list
            item_path = os.path.join(root, item)
            if not os.path.isdir(item_path):
                print(f"Warning: '{item}' is not a directory, skipping.")
                continue # Skip non-directory items listed

            metadata_path = os.path.join(item_path, 'metadata.txt')
            audio_path = os.path.join(item_path, 'full_song.mp3')
            # Check if the specific song from the list is valid
            if os.path.exists(metadata_path) and os.path.exists(audio_path):
                self.song_list.append(item) # Append valid song ID to the attribute
            else:
                 print(f"Warning: Song {item} missing metadata/audio, skipping.")
        # --- END CORRECTION ---

        print(f"Dataset initialized with {len(self.song_list)} valid songs. Max frames: {self.max_frames}.")

    def __len__(self):
        # This should now work as self.song_list is guaranteed to be initialized
        return len(self.song_list)

    def __getitem__(self, idx):
        song = self.song_list[idx] # Get song ID from the initialized list
        metadata_path = os.path.join(self.root, song, 'metadata.txt')
        audio_path = os.path.join(self.root, song, 'full_song.mp3')
        cache_path = os.path.join(self.cache_dir, f'{song}_mel_{self.max_frames}f.pt')

        try:
            mel_db, duration = None, None
            if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
                try:
                    mel_db, duration = torch.load(cache_path)
                    if not isinstance(mel_db, torch.Tensor) or mel_db.shape != (self.n_mels, self.max_frames):
                        print(f"Warning: Invalid cache shape {mel_db.shape} for {song}. Recalculating.")
                        mel_db = None
                    if not isinstance(duration, float): duration = None
                except Exception as e:
                    print(f"Warning: Cache load error for {song}: {e}. Recalculating.")
                    mel_db, duration = None, None

            if mel_db is None or duration is None:
                y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                original_duration = librosa.get_duration(y=y, sr=sr)
                if original_duration is None or original_duration < 1.0: return None

                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, hop_length=self.hop_length)
                mel_db_full = torch.tensor(librosa.power_to_db(mel, ref=np.max), dtype=torch.float32)
                if self.transform: mel_db_full = self.transform(mel_db_full)

                n_frames_original = mel_db_full.shape[1]
                if n_frames_original < self.max_frames:
                    pad_width = self.max_frames - n_frames_original
                    mel_db = nn.functional.pad(mel_db_full, (0, pad_width), mode='constant', value=0)
                else:
                    mel_db = mel_db_full[:, :self.max_frames]

                duration = original_duration # Use original duration for normalization calc later
                try:
                    if mel_db.shape == (self.n_mels, self.max_frames):
                         torch.save((mel_db, duration), cache_path)
                    else:
                         print(f"Error: Invalid shape {mel_db.shape} before saving cache for {song}.")
                except Exception as e:
                    print(f"Error saving cache {cache_path}: {e}")

            if mel_db is None or duration is None or duration <= 0 or mel_db.shape != (self.n_mels, self.max_frames):
                print(f"Error: Invalid final data for {song}. Shape: {mel_db.shape if mel_db is not None else 'None'}. Skipping.")
                return None

            with open(metadata_path, 'r', encoding='utf-8') as f:
                meta = dict(line.strip().split(': ', 1) for line in f if ': ' in line)
            if 'Detected Timestamp' not in meta:
                print(f"Warning: 'Detected Timestamp' missing for {song}. Skipping.")
                return None
            try:
                highlight_start = float(meta['Detected Timestamp'].split()[0])
            except ValueError:
                print(f"Warning: Could not parse timestamp for {song}. Skipping.")
                return None

            highlight_start = max(0.0, min(highlight_start, duration - 0.1 if duration > 0.1 else duration))
            highlight_start_normalized = highlight_start / duration

            # Return 2D tensor (n_mels, max_frames)
            return mel_db, torch.tensor(highlight_start_normalized, dtype=torch.float32), torch.tensor(duration, dtype=torch.float32)

        except Exception as e:
            print(f"--- Error processing {song} in __getitem__ ---\n{traceback.format_exc()}\n--- End Error ---")
            return None

# --- Custom Collate Function ---
def collate_spectrograms(batch):
    # Filter out None items and check for 2D tensors
    batch = [item for item in batch if item is not None and isinstance(item, tuple) and len(item) == 3 and
             isinstance(item[0], torch.Tensor) and item[0].ndim == 2] # Expect (n_mels, max_frames)

    if not batch: return None, None, None
    try:
        # Stacking (n_mels, max_frames) tensors results in (Batch, n_mels, max_frames)
        spectrograms = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])
        durations = torch.stack([item[2] for item in batch])
        return spectrograms, targets, durations
    except Exception as e:
        print(f"Error during collation: {e}")
        # Print shapes to debug if stacking fails due to size mismatch
        for i, item in enumerate(batch):
            if item is not None and isinstance(item, tuple) and len(item) > 0 and isinstance(item[0], torch.Tensor):
                 print(f" Item {i} shape: {item[0].shape}")
        return None, None, None

# --- AST Model ---
class ASTHighlightRegressor(nn.Module):
    def __init__(self, model_name_or_path, dropout, n_mels, max_frames): # Keep n_mels, max_frames for potential config override if needed
        super().__init__()
        try:
            config = ASTConfig.from_pretrained(model_name_or_path)
            # You could potentially override config details here if needed, e.g.:
            # config.num_mel_bins = n_mels
            # config.max_length = max_frames # Be careful overriding this for pretrained models
            self.ast = ASTModel.from_pretrained(model_name_or_path, config=config)
        except Exception as e:
             print(f"Error loading AST model '{model_name_or_path}': {e}")
             raise # Re-raise the exception to stop execution if model load fails

        embed_dim = config.hidden_size
        self.regressor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Input x expected shape: (Batch, n_mels, max_frames)
        # ASTModel expects (Batch, Frequency, Time)
        outputs = self.ast(input_values=x)
        # Use mean pooling over the sequence dimension (dim=1) of last_hidden_state
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        return self.regressor(pooled_output).squeeze(-1) # Output shape: (Batch,)

# --- Evaluation Function ---
def evaluate_model(model, loader, device, loss_fn):
    model.eval()
    total_mae_sec, total_loss, total_samples = 0.0, 0.0, 0
    with torch.no_grad():
        for batch_data in loader:
            spectrograms, targets, durations = batch_data
            if spectrograms is None: continue # Skip empty batches

            spectrograms, targets, durations = spectrograms.to(device), targets.to(device), durations.to(device)
            pred_norm = model(spectrograms)
            loss = loss_fn(pred_norm, targets)

            # Calculate MAE in seconds
            pred_sec = pred_norm * durations
            target_sec = targets * durations
            batch_mae_sec = torch.abs(pred_sec - target_sec)

            # Accumulate loss and MAE
            total_loss += loss.item() * targets.size(0) # loss is already mean over batch, scale back up
            total_mae_sec += batch_mae_sec.sum().item() # Sum MAE across the batch
            total_samples += targets.size(0) # Count number of samples processed

    avg_mae_sec = total_mae_sec / total_samples if total_samples > 0 else 0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    return avg_mae_sec, avg_loss

# --- Training Function ---
def train_model(config):
    run_name = f"{config['run_name_prefix']}_{datetime.now().strftime('%Y%m%d-%H%M')}" # Removed LR/BS from name for brevity
    print(f"\n--- Starting Training Run: {run_name} ---")
    print(f"Using Configuration:\n{config}")

    # --- Data Splitting ---
    all_song_ids = [d for d in os.listdir(config['dataset_folder']) if os.path.isdir(os.path.join(config['dataset_folder'], d))]
    if not all_song_ids:
        raise FileNotFoundError(f"No subdirectories found in dataset folder: {config['dataset_folder']}")
    random.shuffle(all_song_ids)
    total = len(all_song_ids)
    train_cut, val_cut = int(0.8 * total), int(0.9 * total)
    train_song_ids, val_song_ids = all_song_ids[:train_cut], all_song_ids[train_cut:val_cut]
    print(f"Dataset Split - Total: {total}, Train: {len(train_song_ids)}, Val: {len(val_song_ids)}")

    # --- Datasets and Loaders ---
    transform = NormalizeSpectrogram()
    # Pass the split lists of song IDs to the Dataset constructor
    train_ds = FullSpectrogramDataset(config['dataset_folder'], train_song_ids, config['cache_dir'], transform, config['max_frames'], config['n_mels'], config['sample_rate'], config['hop_length_mel'])
    val_ds = FullSpectrogramDataset(config['dataset_folder'], val_song_ids, config['cache_dir'], transform, config['max_frames'], config['n_mels'], config['sample_rate'], config['hop_length_mel'])

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("Error: Training or validation dataset is empty after filtering. Check dataset structure and paths.")
        return

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_spectrograms, num_workers=config['num_workers'], pin_memory=config['pin_memory'], persistent_workers=True if config['num_workers'] > 0 else False)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_spectrograms, num_workers=config['num_workers'], pin_memory=config['pin_memory'], persistent_workers=True if config['num_workers'] > 0 else False)

    # --- Model, Optimizer, Scheduler, Criterion ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = ASTHighlightRegressor(
        model_name_or_path=config['ast_model_name'],
        dropout=config['dropout'],
        n_mels=config['n_mels'],
        max_frames=config['max_frames']
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5, verbose=False) # Quieter scheduler
    criterion = nn.L1Loss()

    # --- Logging Setup ---
    save_dir = os.path.join(config['log_dir'], run_name)
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "results.txt")
    plot_path = os.path.join(save_dir, "plot.png")
    best_model_path = os.path.join(save_dir, "best_model.pth")

    # Save config and model structure
    with open(os.path.join(save_dir, "config_and_model.txt"), "w") as f:
        f.write(f"Run Name: {run_name}\n--- Config ---\n")
        for key, val in config.items(): f.write(f"{key}: {val}\n")
        f.write("\n--- Model Architecture ---\n")
        f.write(str(model))

    # --- Training Loop ---
    train_losses, val_losses, val_maes = [], [], []
    best_val_mae, best_epoch = float('inf'), -1
    start_time_train = time.time()

    try:
        for epoch in range(1, config['epochs'] + 1):
            epoch_start_time = time.time()
            model.train()
            total_train_loss_epoch, processed_batches = 0.0, 0

            for batch_idx, batch_data in enumerate(train_loader):
                spectrograms, targets, durations = batch_data
                if spectrograms is None: # Skip if collation failed
                    print(f"Warning: Skipping empty/invalid batch {batch_idx+1} in epoch {epoch} (train).")
                    continue

                spectrograms, targets = spectrograms.to(device), targets.to(device)

                optimizer.zero_grad()
                y_pred = model(spectrograms) # Forward pass
                loss = criterion(y_pred, targets)

                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected epoch {epoch}, batch {batch_idx+1}. Skipping update.")
                    continue # Skip backward/step

                loss.backward()
                # Optional: Gradient Clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_train_loss_epoch += loss.item()
                processed_batches += 1

            if processed_batches == 0:
                 print(f"Epoch {epoch}/{config['epochs']} - No batches processed in training, skipping evaluation.")
                 continue

            avg_train_loss = total_train_loss_epoch / processed_batches

            # --- Validation ---
            val_mae, avg_val_loss = evaluate_model(model, val_loader, device, criterion)

            # --- Log results ---
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_maes.append(val_mae)

            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch {epoch}/{config['epochs']} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val MAE: {val_mae:.2f}s | Time: {epoch_duration:.2f}s | LR: {optimizer.param_groups[0]['lr']:.1e}")

            # --- Scheduler Step ---
            scheduler.step(avg_val_loss) # Step based on validation loss

            # --- Save Best Model ---
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch
                try:
                    torch.save(model.state_dict(), best_model_path)
                    print(f"  -> Saved new best model (MAE: {best_val_mae:.2f}s at epoch {best_epoch})")
                except Exception as e:
                    print(f"Error saving best model: {e}")

    except KeyboardInterrupt:
        print("\n--- Training interrupted by user ---")
    except Exception as e:
        print(f"\n--- An error occurred during training ---\n{traceback.format_exc()}\n--- End Error ---")
    finally:
        # --- Post-Training Actions ---
        total_train_time = time.time() - start_time_train
        print(f"\nTraining finished. Total time: {total_train_time:.2f} seconds.")
        if best_epoch != -1:
            print(f"Best Validation MAE: {best_val_mae:.2f} seconds (Epoch {best_epoch}) saved to {best_model_path}")
        else:
            print("No best model saved (validation MAE did not improve).")

        # --- Plotting ---
        if train_losses and val_losses and val_maes: # Check if lists are populated
            try:
                plt.figure(figsize=(10, 4))
                epochs_range = range(1, len(train_losses) + 1)
                plt.subplot(1, 2, 1); plt.plot(epochs_range, train_losses, '.-'); plt.plot(epochs_range, val_losses, '.-'); plt.title('Loss (L1)'); plt.grid(True); plt.legend(['Train', 'Val'])
                plt.subplot(1, 2, 2); plt.plot(epochs_range, val_maes, '.-', color='orange'); plt.title('Validation MAE (s)'); plt.grid(True)
                if best_epoch != -1: plt.scatter([best_epoch], [best_val_mae], color='red', s=100, label=f'Best ({best_val_mae:.2f}s)', zorder=5); plt.legend()
                plt.tight_layout(); plt.savefig(plot_path); plt.close()
                print(f"Plot saved to {plot_path}")
            except Exception as e:
                 print(f"Error generating plot: {e}")
        else:
            print("Skipping plot generation as no training/validation results were recorded.")

        # --- Final Log File ---
        try:
            with open(log_file, "a") as f: # Append mode
                f.write(f"\n--- Run End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                if train_losses: f.write(f"Final Train Loss: {train_losses[-1]:.4f}\n")
                if val_losses: f.write(f"Final Val Loss: {val_losses[-1]:.4f}\n")
                if val_maes: f.write(f"Final Val MAE(s): {val_maes[-1]:.2f}\n")
                f.write(f"Best Val MAE(s): {best_val_mae:.2f} at Epoch {best_epoch}\n")
                f.write(f"Total Time: {total_train_time:.2f}s\n")
                f.write(f"Config used: {config}\n") # Log the config used
        except Exception as e:
             print(f"Error writing to log file: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Basic check if dataset folder exists
    if not os.path.isdir(CONFIG['dataset_folder']):
        print(f"Error: Dataset folder not found at {CONFIG['dataset_folder']}")
    else:
        train_model(CONFIG)

    print("\n--- Script Finished ---")