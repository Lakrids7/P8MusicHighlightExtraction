# <<< Keep all imports the same as before >>>
import torch
import torch.nn as nn
import numpy as np
import librosa
import os
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence # Removed pack/pad sequence as LSTM is gone
import random
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
import traceback # Import traceback for detailed error logging
import math # Import math for positional encoding

# --- Constants (can be overridden by argparse) ---
# <<< Keep Constants the same >>>
SAMPLE_RATE = 22050
N_MELS = 128
FRAMES_PER_CHUNK = 128 # How many spectrogram frames per chunk
HOP_LENGTH_MEL = 512 # Default hop length for librosa melspectrogram
CHUNK_DURATION_S_DEFAULT = (FRAMES_PER_CHUNK * HOP_LENGTH_MEL) / SAMPLE_RATE # Approx seconds per chunk (default)
MAX_SEQ_LEN = 512 # Max number of chunks per song (limits memory/compute) - tune this! Needs to be long enough for most songs.


# --- Data Transformation ---
# <<< Keep NormalizeSpectrogram class the same >>>
class NormalizeSpectrogram(nn.Module):
    def forward(self, mel_db):
        mean = mel_db.mean()
        std = mel_db.std()
        if std == 0:
            return mel_db - mean
        # Add epsilon for numerical stability
        return (mel_db - mean) / (std + 1e-9)

# --- Dataset ---
# <<< Keep ChunkedSongDataset class the same >>>
# (It correctly returns chunked tensor, normalized target, and duration)
class ChunkedSongDataset(Dataset):
    def __init__(self, root, song_list=None, cache_dir="mel_cache", transform=None, chunk_frames=FRAMES_PER_CHUNK, max_seq_len=MAX_SEQ_LEN):
        self.root = root
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        all_items = os.listdir(root)
        self.song_list = []
        if song_list is None:
             for item in all_items:
                 item_path = os.path.join(root, item)
                 if os.path.isdir(item_path):
                     metadata_path = os.path.join(item_path, 'metadata.txt')
                     audio_path = os.path.join(item_path, 'full_song.mp3')
                     if os.path.exists(metadata_path) and os.path.exists(audio_path):
                         self.song_list.append(item)
        else:
             for song in song_list:
                 item_path = os.path.join(root, song)
                 metadata_path = os.path.join(item_path, 'metadata.txt')
                 audio_path = os.path.join(item_path, 'full_song.mp3')
                 if os.path.isdir(item_path) and os.path.exists(metadata_path) and os.path.exists(audio_path):
                     self.song_list.append(song)
                 else:
                     print(f"Warning: Song {song} from provided list not found or incomplete in {root}, skipping.")

        self.transform = transform or NormalizeSpectrogram()
        self.chunk_frames = chunk_frames
        self.max_seq_len = max_seq_len
        print(f"Dataset: {len(self.song_list)} valid songs found. Chunk frames: {self.chunk_frames}. Max sequence length: {self.max_seq_len} chunks.")

    def __len__(self):
        return len(self.song_list)

    def __getitem__(self, idx):
        song = self.song_list[idx]
        metadata_path = os.path.join(self.root, song, 'metadata.txt')
        audio_path = os.path.join(self.root, song, 'full_song.mp3')
        cache_path = os.path.join(self.cache_dir, f'{song}_full_mel.pt')

        try:
            mel_db = None
            duration = None

            if os.path.exists(cache_path):
                try:
                    if os.path.getsize(cache_path) > 0:
                        mel_db, duration = torch.load(cache_path)
                        if not isinstance(mel_db, torch.Tensor):
                             print(f"Warning: Invalid mel_db type loaded from cache for {song}. Recalculating.")
                             mel_db = None
                        if not isinstance(duration, float):
                             print(f"Warning: Invalid duration type loaded from cache for {song}. Recalculating.")
                             duration = None
                    else:
                        print(f"Warning: Cache file {cache_path} is empty. Recalculating.")
                except Exception as e:
                    print(f"Warning: Error loading cache {cache_path}: {e}. Recalculating.")
                    mel_db = None
                    duration = None

            if mel_db is None or duration is None:
                y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
                duration = librosa.get_duration(y=y, sr=sr)
                if duration is None or duration < 1.0:
                    print(f"Warning: Song {song} has duration < 1.0s ({duration}). Skipping.")
                    return None

                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH_MEL)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                mel_db = torch.tensor(mel_db, dtype=torch.float32)

                if self.transform:
                    mel_db = self.transform(mel_db)

                try:
                     if isinstance(mel_db, torch.Tensor) and isinstance(duration, float):
                         torch.save((mel_db, duration), cache_path)
                     else:
                         print(f"Error: Invalid data types before saving cache for {song}. Not saving.")
                except Exception as e:
                     print(f"Error saving cache {cache_path}: {e}")

            if mel_db is None or duration is None or duration <= 0:
                 print(f"Error: Could not obtain valid mel spectrogram or duration for {song}. Skipping.")
                 return None

            with open(metadata_path, 'r', encoding='utf-8') as f:
                meta = dict(line.strip().split(': ', 1) for line in f if ': ' in line)

            if 'Detected Timestamp' not in meta:
                 print(f"Warning: Metadata missing 'Detected Timestamp' for song {song}. Skipping.")
                 return None

            try:
                highlight_start = float(meta['Detected Timestamp'].split()[0])
            except ValueError:
                print(f"Warning: Could not parse 'Detected Timestamp' for song {song}: '{meta['Detected Timestamp']}'. Skipping.")
                return None

            highlight_start = max(0.0, min(highlight_start, duration - 0.1 if duration > 0.1 else duration))
            if duration <= 0:
                print(f"Error: Invalid duration {duration} encountered for {song} before normalization. Skipping.")
                return None
            highlight_start_normalized = highlight_start / duration

            n_frames_total = mel_db.shape[1]
            num_chunks = n_frames_total // self.chunk_frames

            if num_chunks == 0:
                 print(f"Warning: Song {song} too short to create any chunks ({n_frames_total} frames < {self.chunk_frames}). Skipping.")
                 return None

            mel_chunks = torch.split(mel_db[:, :num_chunks * self.chunk_frames], self.chunk_frames, dim=1)
            if not mel_chunks:
                print(f"Warning: torch.split resulted in no chunks for {song}. Skipping.")
                return None
            mel_chunk_tensor = torch.stack(mel_chunks, dim=0) # (num_chunks, n_mels, chunk_frames)
            mel_chunk_tensor = mel_chunk_tensor.unsqueeze(1) # Add channel dim -> (num_chunks, 1, n_mels, chunk_frames)

            seq_len = mel_chunk_tensor.size(0)
            if seq_len > self.max_seq_len:
                 mel_chunk_tensor = mel_chunk_tensor[:self.max_seq_len]

            if mel_chunk_tensor.ndim != 4:
                print(f"Error: Final tensor shape invalid for {song} ({mel_chunk_tensor.shape}). Skipping.")
                return None

            return mel_chunk_tensor, torch.tensor(highlight_start_normalized, dtype=torch.float32), torch.tensor(duration, dtype=torch.float32)

        except Exception as e:
            print(f"--- Error processing {song} ---")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {e}")
            print(traceback.format_exc())
            print(f"--- End Error Traceback for {song} ---")
            return None

# --- Custom Collate Function ---
# <<< Keep collate_padded_sequences function the same >>>
# (It correctly returns padded sequences, targets, durations, seq_lengths)
def collate_padded_sequences(batch):
    # Filter out None items more robustly
    batch = [item for item in batch if item is not None and isinstance(item, tuple) and len(item) == 3 and
             isinstance(item[0], torch.Tensor) and isinstance(item[1], torch.Tensor) and isinstance(item[2], torch.Tensor) and
             item[0].ndim == 4] # Ensure sequence tensor has correct dimensions

    if not batch:
        return None, None, None, None

    sequences = [item[0] for item in batch]
    targets = torch.stack([item[1] for item in batch])
    durations = torch.stack([item[2] for item in batch]) # Collect durations

    # Ensure sequences list is not empty before proceeding
    if not sequences:
         return None, None, None, None

    try:
        seq_lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
        # Pad sequences
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    except Exception as e:
        print(f"Error during padding: {e}")
        print(f"Sequence shapes causing error: {[s.shape for s in sequences]}")
        return None, None, None, None # Return None if padding fails

    # Return padded sequences, targets, durations, and original sequence lengths
    return padded_sequences, targets, durations, seq_lengths


# --- Positional Encoding ---
# (Helper class or function for sinusoidal positional encoding)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=MAX_SEQ_LEN): # max_len should match max number of chunks
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe) # Register as buffer so it's part of the model state but not trained

    def forward(self, x):
        # x shape: (Batch, SeqLen, d_model)
        # Add positional encoding up to the length of the sequence
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# --- *** NEW MODEL: NAM-LF (pos) adapted for Regression *** ---
class NAMLFHighlightRegressor(nn.Module):
    def __init__(self, input_features=N_MELS, cnn_out_channels=64, cnn_kernel_size=3, pool_size=2,
                 dropout=0.3, attention_dim=128, max_seq_len=MAX_SEQ_LEN): # Added attention_dim
        super().__init__()
        self.input_features = input_features
        self.cnn_out_channels = cnn_out_channels

        # --- CNN Feature Extractor (Same as before) ---
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=cnn_kernel_size, padding=cnn_kernel_size//2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(pool_size),
            nn.Conv2d(16, 32, kernel_size=cnn_kernel_size, padding=cnn_kernel_size//2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(pool_size),
            nn.Conv2d(32, cnn_out_channels, kernel_size=cnn_kernel_size, padding=cnn_kernel_size//2),
            nn.BatchNorm2d(cnn_out_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Global pooling per chunk
        )
        # Output feature dimension per chunk
        cnn_output_flat_size = cnn_out_channels

        # --- Positional Encoding ---
        self.pos_encoder = PositionalEncoding(cnn_output_flat_size, dropout, max_len=max_seq_len)

        # --- Attention Mechanism (Operates on CNN features + Pos Encoding) ---
        # Feature dimension after adding positional encoding is still cnn_output_flat_size
        feature_dim_with_pos = cnn_output_flat_size

        # Simple attention mechanism (Linear -> Tanh -> Linear -> Softmax (applied later))
        # Similar structure to the paper's attention, adjust attention_dim if needed
        self.attention_fc1 = nn.Linear(feature_dim_with_pos, attention_dim)
        self.attention_fc2 = nn.Linear(attention_dim, 1)
        # Consider adding Tanh activation between fc1 and fc2 as in the paper/previous model
        # self.attention = nn.Sequential(
        #     nn.Linear(feature_dim_with_pos, attention_dim),
        #     nn.Tanh(),
        #     nn.Linear(attention_dim, 1)
        # )


        # --- Regressor (Operates on attention-weighted context vector + duration) ---
        # Input size is feature dimension + 1 (for duration)
        regressor_input_size = feature_dim_with_pos + 1
        self.regressor = nn.Sequential(
            nn.Linear(regressor_input_size, regressor_input_size // 2), # Input size adjusted
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(regressor_input_size // 2, 1),
            nn.Sigmoid() # Output between 0 and 1 (normalized timestamp)
        )

    def forward(self, x, seq_lengths, durations):
        # x shape: (Batch, SeqLen, Channels=1, N_MELS, FRAMES_PER_CHUNK)
        # seq_lengths shape: (Batch,)
        # durations shape: (Batch,)
        batch_size, seq_len, channels, n_mels, frames_per_chunk = x.size()

        # 1. CNN Feature Extraction per Chunk
        x_reshaped = x.view(batch_size * seq_len, channels, n_mels, frames_per_chunk)
        cnn_features = self.cnn(x_reshaped) # Shape: (Batch*SeqLen, cnn_out_channels, 1, 1)
        cnn_features_flat = cnn_features.view(batch_size, seq_len, -1) # Shape: (Batch, SeqLen, cnn_out_channels)

        # 2. Add Positional Encoding
        features_with_pos = self.pos_encoder(cnn_features_flat) # Shape: (Batch, SeqLen, cnn_out_channels)

        # 3. Attention Mechanism
        # Calculate attention scores (unnormalized)
        # Option 1: Using separate layers
        attention_hidden = torch.tanh(self.attention_fc1(features_with_pos)) # (Batch, SeqLen, attention_dim)
        attention_scores = self.attention_fc2(attention_hidden).squeeze(-1)   # (Batch, SeqLen)
        # Option 2: Using sequential (if defined that way)
        # attention_scores = self.attention(features_with_pos).squeeze(-1) # (Batch, SeqLen)


        # Create mask based on sequence lengths
        mask = torch.arange(seq_len, device=x.device)[None, :] < seq_lengths[:, None]
        attention_scores = attention_scores.masked_fill(~mask, float('-inf'))

        # Calculate attention weights (softmax over sequence dimension)
        attention_weights = torch.softmax(attention_scores, dim=1) # Shape: (Batch, SeqLen)

        # 4. Calculate Context Vector (Weighted sum of features_with_pos)
        context_vector = torch.sum(features_with_pos * attention_weights.unsqueeze(-1), dim=1) # Shape: (Batch, cnn_out_channels)

        # 5. Concatenate Duration
        duration_feature = durations.unsqueeze(1) # Shape: (Batch, 1)
        combined_features = torch.cat((context_vector, duration_feature), dim=1) # Shape: (Batch, cnn_out_channels + 1)

        # 6. Regressor
        timestamp = self.regressor(combined_features).squeeze(-1) # Shape: (Batch)
        return timestamp


# --- Training Function ---
# <<< Modified to use the new model class name and potentially different args >>>
def train_model(dataset_folder, epochs, batch_size, lr, model_class, weight_decay,
                 dropout, args): # Removed LSTM params, pass args for logging

    # --- Setup local logging and saving directories ---
    # <<< Keep logging setup the same >>>
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    save_dir = os.path.join("training_logs", run_id)
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "results.txt")
    plot_path = os.path.join(save_dir, "loss_mae_plot.png")
    final_model_path = os.path.join(save_dir, "final_weights.pth")
    best_model_path = os.path.join(save_dir, "best_val_weights.pth")
    model_summary_path = os.path.join(save_dir, "model_architecture.txt")
    print(f"Training run started. Logs and checkpoints will be saved in: {save_dir}")


    # --- Data Loading ---
    # <<< Keep data splitting and loader creation the same >>>
    all_songs = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]
    random.shuffle(all_songs)
    total = len(all_songs)
    train_cut = int(0.8 * total)
    val_cut = int(0.9 * total)
    train_songs = all_songs[:train_cut]
    val_songs = all_songs[train_cut:val_cut]
    print(f"Total songs: {total}, Train songs: {len(train_songs)}, Val songs: {len(val_songs)}")
    train_ds = ChunkedSongDataset(dataset_folder, train_songs, chunk_frames=args.chunk_frames, max_seq_len=args.max_seq_len)
    val_ds = ChunkedSongDataset(dataset_folder, val_songs, chunk_frames=args.chunk_frames, max_seq_len=args.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_padded_sequences, num_workers=4, pin_memory=True, persistent_workers=True if args.batch_size > 0 else False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_padded_sequences, num_workers=4, pin_memory=True, persistent_workers=True if args.batch_size > 0 else False)


    # --- Model, Optimizer, Loss ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # *** Instantiate the NEW model ***
    model = model_class(
        input_features=N_MELS, # From constants
        cnn_out_channels=args.cnn_out_channels, # Add this as an arg if needed
        dropout=dropout,
        max_seq_len=args.max_seq_len # Pass max_seq_len for PositionalEncoding
    ).to(device)

    print(f"Using model: {model_class.__name__}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=10, verbose=True)
    criterion = nn.L1Loss() # MAE Loss

    # <<< Keep model summary saving the same >>>
    with open(model_summary_path, "w") as f:
        f.write(f"Training Run ID: {run_id}\n")
        f.write(f"Model type: {model_class.__name__}\n\n")
        f.write("--- Hyperparameters ---\n")
        for key, val in vars(args).items():
             f.write(f"{key}: {val}\n")
        f.write("\n--- Model Architecture ---\n")
        f.write(str(model))

    # --- Training Loop ---
    train_losses = []
    val_maes = []
    val_losses = []
    best_val_mae = float('inf')
    best_epoch = -1

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            processed_batches = 0
            for batch_idx, batch_data in enumerate(train_loader):
                padded_sequences, targets, durations, seq_lengths = batch_data
                if padded_sequences is None: continue # Skip empty batches

                padded_sequences = padded_sequences.to(device)
                targets = targets.to(device)
                durations = durations.to(device)
                seq_lengths = seq_lengths.to(device)

                optimizer.zero_grad()
                # *** Pass durations to the new model ***
                y_pred = model(padded_sequences, seq_lengths, durations)
                loss = criterion(y_pred, targets)

                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected in epoch {epoch}, batch {batch_idx+1}. Skipping batch update.")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                processed_batches += 1

            if processed_batches == 0:
                 print(f"Epoch {epoch}/{epochs} - No batches processed, skipping.")
                 continue

            avg_train_loss = total_loss / processed_batches

            # Evaluate
            val_mae, avg_val_loss = evaluate_model(model, val_loader, device, criterion)
            val_maes.append(val_mae)
            val_losses.append(avg_val_loss)
            train_losses.append(avg_train_loss)

            print(f"Epoch {epoch}/{epochs} - Train Loss (L1): {avg_train_loss:.4f}, Val Loss (L1): {avg_val_loss:.4f}, Val MAE: {val_mae:.2f}s")

            scheduler.step(avg_val_loss)

            # Save best model
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch
                state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(state_dict, best_model_path)
                print(f"  -> Saved new best model with Val MAE: {best_val_mae:.2f}s")

        # --- Post-Training ---
        # <<< Keep final model saving, plotting, and result logging the same >>>
        # Save final model
        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state_dict, final_model_path)

        # Plotting
        plt.figure(figsize=(12, 5))
        epochs_range = range(1, len(train_losses) + 1)
        plt.subplot(1, 2, 1)
        if train_losses: plt.plot(epochs_range, train_losses, label='Train Loss (L1)', marker='.')
        if val_losses: plt.plot(epochs_range, val_losses, label='Validation Loss (L1)', marker='.')
        plt.xlabel('Epoch'); plt.ylabel('Loss (L1)'); plt.title('Training & Validation Loss'); plt.legend(); plt.grid(True)
        plt.subplot(1, 2, 2)
        if val_maes:
            plt.plot(epochs_range, val_maes, label='Validation MAE (s)', color='orange', marker='.')
            if best_epoch != -1: plt.scatter([best_epoch], [best_val_mae], color='red', s=100, label=f'Best MAE ({best_val_mae:.2f}s)', zorder=5)
        plt.xlabel('Epoch'); plt.ylabel('MAE (seconds)'); plt.title('Validation Mean Absolute Error'); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(plot_path); plt.close()

        # Logging results
        with open(log_file, "a") as f:
            f.write("\n--- Final Results ---\n")
            if train_losses: f.write(f"Final Training Loss (L1): {train_losses[-1]:.4f}\n")
            if val_losses: f.write(f"Final Validation Loss (L1): {val_losses[-1]:.4f}\n")
            if val_maes: f.write(f"Final Validation MAE (seconds): {val_maes[-1]:.2f}\n")
            f.write(f"Best Validation MAE (seconds): {best_val_mae:.2f} (Epoch {best_epoch})\n")
            f.write(f"Final model saved to: {final_model_path}\n")
            f.write(f"Best model saved to: {best_model_path}\n")
            f.write(f"Loss plot saved to: {plot_path}\n")
            f.write(f"Model summary saved to: {model_summary_path}\n")

        print(f"\nTraining complete. Logs and models saved locally to {save_dir}")
        print(f"Best Validation MAE achieved: {best_val_mae:.2f} seconds (Epoch {best_epoch}).")

    except Exception as e:
        print(f"\n--- An error occurred during training ---")
        print(traceback.format_exc())
        with open(log_file, "a") as f:
             f.write("\n--- Error during training ---\n")
             f.write(traceback.format_exc())


# --- Evaluation Function ---
# <<< Modified to pass duration to the new model >>>
def evaluate_model(model, loader, device, loss_fn):
    model.eval()
    total_mae_sec = 0.0
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch_data in loader:
            padded_sequences, targets, durations, seq_lengths = batch_data
            if padded_sequences is None: continue

            padded_sequences = padded_sequences.to(device)
            targets = targets.to(device)
            durations = durations.to(device)
            seq_lengths = seq_lengths.to(device)

            # *** Pass durations to the new model ***
            pred_norm = model(padded_sequences, seq_lengths, durations)

            # Calculate L1 loss
            loss = loss_fn(pred_norm, targets)

            # Calculate MAE in seconds
            pred_sec = pred_norm * durations
            target_sec = targets * durations
            batch_mae_sec = torch.abs(pred_sec - target_sec)

            total_loss += loss.item() * targets.size(0)
            total_mae_sec += batch_mae_sec.sum().item()
            total_samples += targets.size(0)

    avg_mae_sec = total_mae_sec / total_samples if total_samples > 0 else 0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    return avg_mae_sec, avg_loss


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train highlight regression model (NAM-LF pos Architecture)")
    # <<< Keep dataset, epochs, batch_size, lr, weight_decay args the same >>>
    parser.add_argument('--dataset_folder', type=str, default='Filtered_Songs2', help="Path to the dataset folder")
    parser.add_argument('--epochs', type=int, default=500, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay for Adam optimizer")

    # <<< Updated model name/selection, removed LSTM args, added CNN args maybe >>>
    # parser.add_argument('--model_name', type=str, default='namlf_pos', help="Model name ('namlf_pos')") # Set default
    parser.add_argument('--cnn_out_channels', type=int, default=64, help="Output channels of the last CNN layer") # Make CNN size configurable
    parser.add_argument('--chunk_frames', type=int, default=FRAMES_PER_CHUNK, help="Number of spectrogram frames per chunk")
    parser.add_argument('--max_seq_len', type=int, default=MAX_SEQ_LEN, help="Maximum number of chunks per sequence (for positional encoding)")
    parser.add_argument('--dropout', type=float, default=0.3, help="Dropout rate")
    # Removed --lstm_hidden_size, --lstm_layers

    args = parser.parse_args()

    # Define the model class to use
    model_class = NAMLFHighlightRegressor # Use the new model

    if not os.path.isdir(args.dataset_folder):
        raise FileNotFoundError(f"Dataset folder not found: {args.dataset_folder}")

    print("\n--- Starting Training (NAM-LF pos Architecture) ---")
    print(f"Configuration: {vars(args)}")

    actual_chunk_duration_s = (args.chunk_frames * HOP_LENGTH_MEL) / SAMPLE_RATE
    print(f"Using Chunk Frames: {args.chunk_frames}, Approx Chunk Duration: {actual_chunk_duration_s:.3f}s")
    print(f"Using Max Sequence Length: {args.max_seq_len} chunks")

    # Call the training function with updated arguments
    train_model(
        dataset_folder=args.dataset_folder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_class=model_class,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        args=args # Pass the args object itself
    )

    print("\n--- Training Script Finished ---")