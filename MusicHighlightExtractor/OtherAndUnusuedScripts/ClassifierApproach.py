import torch
import torch.nn as nn
import numpy as np
import librosa
import os
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
import random
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from torchvision.models import resnet18 # Keep if you want to try resnet later

# --- Constants ---
SAMPLE_RATE = 22050
N_MELS = 128
# Define chunk parameters (tune these)
FRAMES_PER_CHUNK = 128 # How many spectrogram frames per chunk
HOP_LENGTH_MEL = 512 # Default hop length for librosa melspectrogram
CHUNK_DURATION_S = (FRAMES_PER_CHUNK * HOP_LENGTH_MEL) / SAMPLE_RATE # Approx seconds per chunk
MAX_SEQ_LEN = 512 # Max number of chunks per song (limits memory/compute) - tune this! Needs to be long enough for most songs.

# --- Data Transformation ---
# Normalization applied per song
class NormalizeSpectrogram(nn.Module):
    def forward(self, mel_db):
        # Normalize the entire song's spectrogram before chunking
        return (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)

# --- Dataset ---
class ChunkedSongDataset(Dataset):
    def __init__(self, root, song_list=None, cache_dir="mel_cache", transform=None, chunk_frames=FRAMES_PER_CHUNK, max_seq_len=MAX_SEQ_LEN):
        self.root = root
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.song_list = song_list or [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        self.transform = transform or NormalizeSpectrogram()
        self.chunk_frames = chunk_frames
        self.max_seq_len = max_seq_len
        print(f"Dataset: {len(self.song_list)} songs. Chunk frames: {self.chunk_frames}. Max sequence length: {self.max_seq_len} chunks.")

    def __len__(self):
        return len(self.song_list)

    def __getitem__(self, idx):
        song = self.song_list[idx]
        metadata_path = os.path.join(self.root, song, 'metadata.txt')
        audio_path = os.path.join(self.root, song, 'full_song.mp3')
        # Cache full spectrogram now
        cache_path = os.path.join(self.cache_dir, f'{song}_full_mel.pt')

        try:
            if os.path.exists(cache_path):
                mel_db, duration = torch.load(cache_path)
                # Ensure it's float32 tensor
                if not isinstance(mel_db, torch.Tensor):
                     mel_db = torch.tensor(mel_db, dtype=torch.float32)
                if not isinstance(duration, float):
                    duration = float(duration)

            else:
                y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
                duration = librosa.get_duration(y=y, sr=sr)
                if duration < 1.0: # Skip very short files
                    print(f"Warning: Song {song} is too short ({duration}s), skipping.")
                    # Return dummy data, will be filtered by collate_fn
                    return None, None, None

                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH_MEL)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                mel_db = torch.tensor(mel_db, dtype=torch.float32) # Convert to tensor
                # Apply normalization to the full spectrogram
                if self.transform:
                    mel_db = self.transform(mel_db)
                torch.save((mel_db, duration), cache_path)

            with open(metadata_path, 'r', encoding='utf-8') as f:
                meta = dict(line.strip().split(': ', 1) for line in f if ': ' in line)
            highlight_start = float(meta['Detected Timestamp'].split()[0])
            # Avoid division by zero for very short durations potentially loaded from cache
            highlight_start_normalized = highlight_start / duration if duration > 0 else 0.0

            # --- Chunking ---
            n_frames_total = mel_db.shape[1]
            # Pad if shorter than one chunk
            if n_frames_total < self.chunk_frames:
                 padding = self.chunk_frames - n_frames_total
                 mel_db = torch.nn.functional.pad(mel_db, (0, padding), mode='constant', value=mel_db.min()) # Pad time axis
                 n_frames_total = self.chunk_frames # Update total frames after padding


            # Split into chunks: (N_MELS, N_FRAMES_TOTAL) -> (N_CHUNKS, N_MELS, FRAMES_PER_CHUNK)
            # Use unfold to create overlapping chunks, or simple splitting for non-overlapping
            # Using simple splitting here for simplicity:
            num_chunks = n_frames_total // self.chunk_frames
            mel_chunks = torch.split(mel_db[:, :num_chunks * self.chunk_frames], self.chunk_frames, dim=1)
            mel_chunk_tensor = torch.stack(mel_chunks, dim=0) # Shape: (num_chunks, N_MELS, chunk_frames)

            # Add channel dimension: (num_chunks, 1, N_MELS, chunk_frames)
            mel_chunk_tensor = mel_chunk_tensor.unsqueeze(1)

            # Truncate if longer than max_seq_len
            seq_len = min(mel_chunk_tensor.size(0), self.max_seq_len)
            mel_chunk_tensor = mel_chunk_tensor[:seq_len]

            return mel_chunk_tensor, torch.tensor(highlight_start_normalized, dtype=torch.float32), torch.tensor(duration, dtype=torch.float32)

        except Exception as e:
            print(f"Error processing {song}: {e}")
            # Return dummy data, will be filtered by collate_fn
            return None, None, None

# --- Custom Collate Function ---
def collate_padded_sequences(batch):
    # Filter out None items resulting from errors
    batch = [item for item in batch if item[0] is not None]
    if not batch: # If all items in batch failed
        return None, None, None, None

    # Separate sequences, targets, and durations
    sequences = [item[0] for item in batch]
    targets = torch.stack([item[1] for item in batch])
    durations = torch.stack([item[2] for item in batch])

    # Get sequence lengths BEFORE padding
    seq_lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)

    # Pad sequences in the batch to the length of the longest sequence IN THIS BATCH
    # We need padding value that makes sense for normalized spectrograms (e.g., 0 or min value)
    # Assuming normalization makes mean close to 0, let's use 0.
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)

    return padded_sequences, targets, durations, seq_lengths

# --- CNN + LSTM Model ---
class CNNLSTMHighlightRegressor(nn.Module):
    def __init__(self, input_features=N_MELS, cnn_out_channels=64, cnn_kernel_size=3, pool_size=2,
                 lstm_hidden_size=128, lstm_layers=2, bidirectional=True, dropout=0.3):
        super().__init__()
        self.input_features = input_features
        self.cnn_out_channels = cnn_out_channels
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional

        # Simple CNN Feature Extractor (applied per chunk)
        # Input: (Batch, 1, N_MELS, FRAMES_PER_CHUNK)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=cnn_kernel_size, padding=cnn_kernel_size//2),
            nn.BatchNorm2d(16), # Added BatchNorm
            nn.ReLU(),
            nn.MaxPool2d(pool_size),
            nn.Conv2d(16, 32, kernel_size=cnn_kernel_size, padding=cnn_kernel_size//2),
            nn.BatchNorm2d(32), # Added BatchNorm
            nn.ReLU(),
            nn.MaxPool2d(pool_size),
            nn.Conv2d(32, cnn_out_channels, kernel_size=cnn_kernel_size, padding=cnn_kernel_size//2),
            nn.BatchNorm2d(cnn_out_channels), # Added BatchNorm
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Output: (Batch, cnn_out_channels, 1, 1)
        )

        # Calculate flattened CNN output size
        cnn_output_flat_size = cnn_out_channels

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=cnn_output_flat_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True, # Crucial: Input shape (batch, seq_len, features)
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0 # Add dropout between LSTM layers
        )

        # Regressor Head
        lstm_output_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout), # Added Dropout
            nn.Linear(lstm_output_size // 2, 1),
            nn.Sigmoid() # Output normalized timestamp
        )

    def forward(self, x, seq_lengths):
        # x shape: (Batch, SeqLen, Channels=1, N_MELS, FRAMES_PER_CHUNK)
        batch_size, seq_len, channels, n_mels, frames_per_chunk = x.size()

        # Apply CNN to each chunk
        # Reshape to treat chunks as batch items for CNN: (Batch * SeqLen, Channels, N_MELS, FRAMES_PER_CHUNK)
        x_reshaped = x.view(batch_size * seq_len, channels, n_mels, frames_per_chunk)
        cnn_features = self.cnn(x_reshaped) # Output: (Batch * SeqLen, cnn_out_channels, 1, 1)
        cnn_features_flat = cnn_features.view(batch_size, seq_len, -1) # Reshape back: (Batch, SeqLen, cnn_out_channels)

        # Pack sequence for LSTM (optional but good practice, handles padding efficiently)
        # Note: pack_padded_sequence requires lengths on CPU
        # packed_input = pack_padded_sequence(cnn_features_flat, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)

        # Feed sequence into LSTM
        # self.lstm.flatten_parameters() # Helpful for DataParallel
        lstm_out, (h_n, c_n) = self.lstm(cnn_features_flat) # Use unpacked input directly if not packing
        # lstm_out shape: (Batch, SeqLen, lstm_hidden_size * num_directions)
        # h_n shape: (num_layers * num_directions, Batch, lstm_hidden_size)

        # Unpack sequence (if packed)
        # lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)

        # --- Option 1: Use the final hidden state ---
        # Concatenate final forward and backward hidden states from the last layer
        if self.bidirectional:
             # h_n is (num_layers * 2, batch, hidden_size)
             # Get last layer's forward and backward states
             final_hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1) # Shape: (Batch, lstm_hidden_size * 2)
        else:
             # h_n is (num_layers, batch, hidden_size)
             final_hidden = h_n[-1,:,:] # Shape: (Batch, lstm_hidden_size)

        # --- Option 2: Use attention or pooling over LSTM outputs (Potentially better) ---
        # Example: Max pooling over time
        # final_hidden, _ = torch.max(lstm_out, dim=1) # Shape: (Batch, lstm_hidden_size * num_directions)
        # Example: Average pooling over time (consider sequence lengths)
        # Need to mask padding before averaging:
        # mask = torch.arange(seq_len, device=x.device)[None, :] < seq_lengths[:, None] # Create mask (Batch, SeqLen)
        # mask = mask.unsqueeze(-1).float() # (Batch, SeqLen, 1)
        # masked_lstm_out = lstm_out * mask
        # sum_lstm_out = torch.sum(masked_lstm_out, dim=1) # (Batch, lstm_hidden_size * num_directions)
        # # Divide by actual sequence length (handle potential zero length)
        # seq_lengths_exp = seq_lengths.unsqueeze(1).float().clamp(min=1) # (Batch, 1)
        # final_hidden = sum_lstm_out / seq_lengths_exp

        # Pass the chosen representation through the regressor
        timestamp = self.regressor(final_hidden).squeeze(-1) # Output: (Batch)
        return timestamp

# --- Training Function (Modified) ---
def train_model(dataset_folder, epochs=50, batch_size=8, lr=1e-4, model_class=CNNLSTMHighlightRegressor, weight_decay=1e-5): # Added weight decay
    all_songs = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]
    random.shuffle(all_songs)
    total = len(all_songs)
    train_cut = int(0.8 * total)
    val_cut = int(0.9 * total)
    train_songs = all_songs[:train_cut]
    val_songs = all_songs[train_cut:val_cut]
    # test_songs = all_songs[val_cut:] # Keep test set separate

    print(f"Train songs: {len(train_songs)}, Val songs: {len(val_songs)}")

    train_ds = ChunkedSongDataset(dataset_folder, train_songs)
    val_ds = ChunkedSongDataset(dataset_folder, val_songs)

    # Use custom collate function
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_padded_sequences, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_padded_sequences, num_workers=4, pin_memory=True) # Use batch_size > 1 for validation too

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model_class().to(device)
    # Consider DataParallel if multiple GPUs
    # if torch.cuda.device_count() > 1:
    #    print(f"Using {torch.cuda.device_count()} GPUs!")
    #    model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # Added weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5, verbose=True) # LR scheduler
    # Use MAE Loss (L1Loss) as it directly matches your evaluation metric
    # criterion = nn.L1Loss()
    criterion = nn.MSELoss() # Or stick with MSE if preferred

    # Set up save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("training_logs", f"{model_class.__name__}_run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "results.txt")
    plot_path = os.path.join(save_dir, "loss_plot.png")
    final_model_path = os.path.join(save_dir, "final_weights.pth")
    best_model_path = os.path.join(save_dir, "best_val_weights.pth")
    model_summary_path = os.path.join(save_dir, "model_architecture.txt")

    # Save model name and full architecture
    with open(model_summary_path, "w") as f:
        f.write(f"Model type: {model_class.__name__}\n")
        f.write(f"Chunk frames: {FRAMES_PER_CHUNK}, Max Seq Len: {MAX_SEQ_LEN}\n")
        f.write(f"Approx Chunk Duration: {CHUNK_DURATION_S:.2f}s\n\n")
        f.write(str(model))
        # Add hyperparameter info
        f.write("\n--- Hyperparameters ---\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Weight Decay: {weight_decay}\n")
        f.write(f"Optimizer: Adam\n")
        f.write(f"Criterion: {criterion.__class__.__name__}\n")


    train_losses = []
    val_maes = []
    val_losses = []
    best_val_mae = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        processed_batches = 0
        for padded_sequences, targets, durations, seq_lengths in train_loader:
            # Skip batch if collate_fn returned None
            if padded_sequences is None:
                continue

            # Move data to device
            padded_sequences, targets = padded_sequences.to(device), targets.to(device)
            seq_lengths = seq_lengths.to(device) # Keep lengths on device if possible

            optimizer.zero_grad()
            # Pass sequence lengths to the model
            y_pred = model(padded_sequences, seq_lengths)
            loss = criterion(y_pred, targets)

            # Handle potential NaN loss
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected in epoch {epoch}. Skipping batch.")
                # Potentially log more info about the batch here
                continue

            loss.backward()
            # Gradient clipping (helps prevent exploding gradients with RNNs)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            processed_batches += 1

        if processed_batches == 0:
            print(f"Epoch {epoch}/{epochs} - No batches processed, skipping epoch.")
            continue

        avg_train_loss = total_loss / processed_batches
        train_losses.append(avg_train_loss)

        val_mae, val_loss = evaluate_model(model, val_loader, device, criterion)
        val_maes.append(val_mae)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f}s")

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            # Save model state_dict (handle DataParallel)
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, best_model_path)
            print(f"  -> Saved new best model with Val MAE: {best_val_mae:.2f}s")


    # Save final model
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state_dict, final_model_path)

    # Plot training loss and validation loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_maes, label='Validation MAE (s)', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (seconds)')
    plt.title('Validation MAE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    # Write log
    with open(log_file, "w") as f:
        f.write(f"Model type: {model_class.__name__}\n")
        f.write(f"Final Training Loss: {train_losses[-1]:.4f}\n")
        f.write(f"Best Validation MAE (seconds): {best_val_mae:.2f}\n")
        f.write(f"Final Validation Loss: {val_losses[-1]:.4f}\n")
        f.write(f"Final Validation MAE (seconds): {val_maes[-1]:.2f}\n")
        f.write(f"Final model path: {final_model_path}\n")
        f.write(f"Best model path: {best_model_path}\n")
        f.write(f"Loss plot: {plot_path}\n")
        f.write(f"Model architecture: {model_summary_path}\n")

    print(f"Training complete. Logs and models saved to {save_dir}")
    print(f"Best Validation MAE achieved: {best_val_mae:.2f} seconds.")


# --- Evaluation Function (Modified) ---
def evaluate_model(model, loader, device, loss_fn):
    model.eval()
    total_mae_sec = 0.0
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for padded_sequences, targets, durations, seq_lengths in loader:
             # Skip batch if collate_fn returned None
            if padded_sequences is None:
                continue

            # Move data to device
            padded_sequences, targets, durations = padded_sequences.to(device), targets.to(device), durations.to(device)
            seq_lengths = seq_lengths.to(device)

            # Get predictions
            pred_norm = model(padded_sequences, seq_lengths) # Normalized prediction
            loss = loss_fn(pred_norm, targets)

            # Calculate MAE in seconds
            pred_sec = pred_norm * durations
            target_sec = targets * durations
            batch_mae_sec = torch.abs(pred_sec - target_sec)

            total_loss += loss.item() * targets.size(0) # Sum loss weighted by batch size
            total_mae_sec += batch_mae_sec.sum().item() # Sum MAE for all samples in batch
            total_samples += targets.size(0) # Count total samples processed

    avg_mae_sec = total_mae_sec / total_samples if total_samples > 0 else 0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    return avg_mae_sec, avg_loss


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train highlight regression model")
    parser.add_argument('--dataset_folder', type=str, default='Filtered_Songs', help="Path to the dataset folder")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training") # Might need to decrease with LSTM
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--model_name', type=str, default='cnnlstm', help="Model name ('cnn', 'cnnlstm')")
    # Add args for model hyperparameters if needed
    parser.add_argument('--chunk_frames', type=int, default=FRAMES_PER_CHUNK, help="Number of spectrogram frames per chunk")
    parser.add_argument('--max_seq_len', type=int, default=MAX_SEQ_LEN, help="Maximum number of chunks per sequence")

    args = parser.parse_args()

    # Update constants based on args
    FRAMES_PER_CHUNK = args.chunk_frames
    MAX_SEQ_LEN = args.max_seq_len

    # Model dictionary
    model_dict = {
        # Keep your original CNN for comparison if needed, but update its Dataset/Dataloader if you run it
        # "cnn": CNNHighlightRegressor,
        "cnnlstm": CNNLSTMHighlightRegressor,
    }

    if args.model_name.lower() not in model_dict:
        raise ValueError(f"Unknown model name: {args.model_name}. Available: {list(model_dict.keys())}")

    model_class = model_dict[args.model_name.lower()]

    # Check dataset folder exists
    if not os.path.isdir(args.dataset_folder):
        raise FileNotFoundError(f"Dataset folder not found: {args.dataset_folder}")

    train_model(
        args.dataset_folder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_class=model_class
    )