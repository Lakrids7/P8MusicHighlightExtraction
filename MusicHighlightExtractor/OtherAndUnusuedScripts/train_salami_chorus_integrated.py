#!/usr/bin/env python3
# train_salami_chorus_integrated.py
import os, json, math, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # For plotting
from tqdm import tqdm # For progress bars

N_MELS, CHUNK_FRAMES, TARGET_DIM, MEL_SUFFIX = 128, 128, 2, ".npy"
DEFAULT_AUGMENTED_CHORUS_CSV = "salami_chorus_annotations_with_duration.csv"
DEFAULT_PRECOMPUTED_MEL_BASE_DIR = "data/salami_mels"
DEFAULT_SPLIT_JSON_PATH = "salami_chorus_split.json"
CHORUS_ANNOTATIONS_DF = None

def load_chorus_annotations_globally(csv_path):
    global CHORUS_ANNOTATIONS_DF
    if CHORUS_ANNOTATIONS_DF is None:
        if not os.path.exists(csv_path): raise FileNotFoundError(f"CSV not found: {csv_path}")
        CHORUS_ANNOTATIONS_DF = pd.read_csv(csv_path)
        CHORUS_ANNOTATIONS_DF['salami_id'] = CHORUS_ANNOTATIONS_DF['salami_id'].astype(str)
        # Basic filtering for valid entries
        CHORUS_ANNOTATIONS_DF = CHORUS_ANNOTATIONS_DF[
            (CHORUS_ANNOTATIONS_DF['total_song_duration_sec'] > 0.1) &
            (CHORUS_ANNOTATIONS_DF['chorus_duration'] > 0.05)
        ]
        if CHORUS_ANNOTATIONS_DF.empty: raise ValueError(f"No valid data after filtering in CSV {csv_path}")
        print(f"Loaded {len(CHORUS_ANNOTATIONS_DF)} chorus entries globally from {csv_path}.")

def get_or_create_data_split(split_json_path, mel_base_dir, force_recreate_split=False, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    if not force_recreate_split and os.path.exists(split_json_path):
        print(f"Loading existing data split from: {split_json_path}")
        try:
            with open(split_json_path, 'r') as f: split_data = json.load(f)
            if all(k in split_data for k in ["train", "val", "test"]) and \
               isinstance(split_data["train"], list) and \
               isinstance(split_data["val"], list) and \
               isinstance(split_data["test"], list):
                return split_data
            print(f"Warning: Malformed or incomplete split file {split_json_path}. Recreating.")
        except json.JSONDecodeError:
            print(f"Warning: Corrupted split file {split_json_path}. Recreating.")
        except Exception as e:
            print(f"Warning: Error reading split file {split_json_path}: {e}. Recreating.")
    return create_new_split(split_json_path, mel_base_dir, train_ratio, val_ratio, test_ratio)

def create_new_split(split_json_path, mel_base_dir, train_ratio, val_ratio, test_ratio):
    print(f"Creating new data split and saving to: {split_json_path}")
    if CHORUS_ANNOTATIONS_DF is None:
        raise RuntimeError("Chorus annotations DataFrame not loaded. Cannot create split.")
    
    # Get unique salami_ids that have corresponding Mel spectrograms
    available_salami_ids = CHORUS_ANNOTATIONS_DF['salami_id'].unique()
    valid_ids_with_mels = []
    for sid in available_salami_ids:
        # Construct the expected path for the precomputed Mel file
        mel_file_path = os.path.join(mel_base_dir, str(sid), f"full_song{MEL_SUFFIX}")
        if os.path.exists(mel_file_path):
            valid_ids_with_mels.append(str(sid))
        # else:
            # print(f"Debug: Mel file not found for ID {sid} at {mel_file_path}")

    if not valid_ids_with_mels:
        raise ValueError(f"No songs with existing Mel spectrograms found in {mel_base_dir} among the annotated songs.")
    print(f"Found {len(valid_ids_with_mels)} unique songs with annotations AND existing Mel spectrograms for splitting.")

    # Ensure ratios sum to 1.0
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.0.")

    # Shuffle the valid IDs
    np.random.seed(42) # for reproducibility
    np.random.shuffle(valid_ids_with_mels)

    # Calculate split indices
    n_total = len(valid_ids_with_mels)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    
    # Handle cases with very few songs robustly
    if n_total < 3: # Not enough for train/val/test
        print("Warning: Very few songs available (<3). Assigning all to train or as available.")
        if n_total == 1:
            train_ids = valid_ids_with_mels
            val_ids, test_ids = [], []
        elif n_total == 2:
            train_ids = [valid_ids_with_mels[0]]
            val_ids = [valid_ids_with_mels[1]] if val_ratio > 0 else []
            test_ids = [valid_ids_with_mels[1]] if test_ratio > 0 and not val_ids else []
            if not val_ids and not test_ids: # if val_ratio and test_ratio are 0
                 val_ids = [valid_ids_with_mels[1]] # assign to val by default
        else: # n_total == 0, should have been caught earlier
            train_ids, val_ids, test_ids = [],[],[]

    else: # Standard split logic
        train_ids = valid_ids_with_mels[:n_train]
        val_ids = valid_ids_with_mels[n_train : n_train + n_val]
        test_ids = valid_ids_with_mels[n_train + n_val:]
        
        # Adjust if rounding caused issues, ensure all IDs are assigned
        # This basic split might leave out a few if n_val or n_test is rounded down.
        # A more robust way is train_test_split twice, but this is simpler for now.
        # For instance, ensure if ratios are non-zero, the splits are non-empty if possible.
        if val_ratio > 0 and not val_ids and test_ids: # If val was supposed to have IDs but got none
            val_ids = [test_ids.pop(0)] if test_ids else []
        if train_ratio > 0 and not train_ids and (val_ids or test_ids):
            if val_ids: train_ids = [val_ids.pop(0)]
            elif test_ids: train_ids = [test_ids.pop(0)]


    # Convert Salami IDs to full paths to the song's Mel directory
    def ids_to_mel_dir_paths(ids_list):
        return [os.path.join(mel_base_dir, str(sid)) for sid in ids_list]

    split_data = {
        "train": ids_to_mel_dir_paths(train_ids),
        "val": ids_to_mel_dir_paths(val_ids),
        "test": ids_to_mel_dir_paths(test_ids)
    }

    try:
        # Ensure the directory for the split JSON exists
        split_json_dir = os.path.dirname(split_json_path)
        if split_json_dir and not os.path.exists(split_json_dir):
            os.makedirs(split_json_dir, exist_ok=True)
            
        with open(split_json_path, 'w') as f:
            json.dump(split_data, f, indent=4)
        print(f"Saved new data split: Train: {len(train_ids)} songs, Val: {len(val_ids)} songs, Test: {len(test_ids)} songs.")
    except IOError as e:
        raise IOError(f"Could not write data split file to {split_json_path}: {e}")
    return split_data


def parse_salami_chorus_labels_and_duration(song_dir_path):
    global CHORUS_ANNOTATIONS_DF
    if CHORUS_ANNOTATIONS_DF is None: return None
    try: salami_id = os.path.basename(song_dir_path)
    except Exception: return None # Should not happen if song_dir_path is valid
    
    song_choruses_df = CHORUS_ANNOTATIONS_DF[CHORUS_ANNOTATIONS_DF['salami_id'] == salami_id]
    if song_choruses_df.empty: return None

    all_norm_choruses = []
    # Initialize total_dur; we'll take it from the first row for this song ID
    # Assuming total_song_duration_sec is consistent per song_id in the CSV
    total_dur = song_choruses_df['total_song_duration_sec'].iloc[0] if not song_choruses_df.empty else -1.0

    for _, chorus_row in song_choruses_df.iterrows():
        start, end = float(chorus_row['chorus_start_time']), float(chorus_row['chorus_end_time'])
        # Optional: Check for consistency in total_song_duration_sec if it varies per row (should not)
        # current_total_dur = float(chorus_row['total_song_duration_sec'])
        # if not math.isclose(total_dur, current_total_dur):
        #     print(f"Warning: Inconsistent total_song_duration_sec for Salami ID {salami_id}. Using first: {total_dur}.")
        
        if total_dur > 1e-6 and end > start: # Use a small epsilon for total_dur
            norm_start = max(0.0, min(1.0, start / total_dur))
            norm_end = max(0.0, min(1.0, end / total_dur))
            if norm_end > norm_start: # Ensure valid normalized segment
                all_norm_choruses.append(np.array([norm_start, norm_end], dtype=np.float32))
    
    if not all_norm_choruses or total_dur <= 1e-6:
        return None
    
    # Return a list of numpy arrays (each [start_norm, end_norm]) and the single total_duration
    return all_norm_choruses, total_dur


class SongDataset(Dataset):
    def __init__(self, song_dirs_from_split):
        self.items = [] # Store (mel_path, song_dir_path) tuples
        if CHORUS_ANNOTATIONS_DF is None:
            raise RuntimeError("Chorus annotations DataFrame not loaded at SongDataset initialization.")
        
        for song_dir_path in song_dirs_from_split:
            mel_path = os.path.join(song_dir_path, f"full_song{MEL_SUFFIX}")
            if os.path.exists(mel_path):
                 # Also verify that annotations exist for this song_id
                 salami_id = os.path.basename(song_dir_path)
                 if not CHORUS_ANNOTATIONS_DF[CHORUS_ANNOTATIONS_DF['salami_id'] == salami_id].empty:
                     self.items.append((mel_path, song_dir_path))
                 # else:
                     # print(f"Debug: No chorus annotations in DF for Salami ID {salami_id} (from dir {song_dir_path}), though Mel exists.")
            # else:
                 # print(f"Debug: Mel file not found at {mel_path} for dir {song_dir_path}")


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        mel_path, song_dir_path = self.items[idx]
        try:
            mel_chunks = np.load(mel_path) # Shape: (num_chunks, N_MELS, CHUNK_FRAMES)
            
            # Ensure Mel chunks have the correct dimensions after loading
            if not (mel_chunks.ndim == 3 and mel_chunks.shape[1] == N_MELS and mel_chunks.shape[2] == CHUNK_FRAMES and mel_chunks.shape[0] > 0):
                # print(f"Warning: Mel chunks at {mel_path} have unexpected shape {mel_chunks.shape}. Expected (N, {N_MELS}, {CHUNK_FRAMES}). Skipping.")
                return None

            parsed_output = parse_salami_chorus_labels_and_duration(song_dir_path)

            if parsed_output is not None:
                label_np_list, duration_float = parsed_output # label_np_list is a list of np.arrays
                
                if not label_np_list: # If parse_salami_chorus_labels returns empty list of choruses
                    # print(f"Warning: No valid chorus labels returned for {song_dir_path}. Skipping.")
                    return None

                labels_tensor = torch.from_numpy(np.array(label_np_list, dtype=np.float32)) # Shape: [num_choruses, 2]

                return (torch.from_numpy(mel_chunks.astype(np.float32)),
                        labels_tensor, 
                        torch.tensor(duration_float, dtype=torch.float32))
            # else:
                # print(f"Warning: Could not parse labels/duration for {song_dir_path}. Skipping.")

        except Exception as e:
            print(f"Error in SongDataset __getitem__ for {mel_path} (song_dir: {song_dir_path}): {e}")
            # import traceback
            # traceback.print_exc()
        return None


def collate_fn(batch):
    # Filter out None items that may have resulted from errors in __getitem__
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None, None, None # Return None for all if batch is empty
    
    mels, labels_list_of_tensors, durations = zip(*batch)
    
    # Pad Mel sequences
    lengths = torch.tensor([mel.shape[0] for mel in mels]) # Number of chunks per song
    padded_mels = pad_sequence(mels, batch_first=True, padding_value=0.0) # Pads along the num_chunks dimension
    
    # Durations can be stacked as they are single floats per item
    durations = torch.stack(durations) 
    
    # labels_list_of_tensors is already a list of tensors (one per song, each [num_choruses, 2])
    # It does not need further processing here if the loss function handles it.
    return padded_mels, list(labels_list_of_tensors), lengths, durations


def positional_encoding(L, D, device): # L=num_chunks, D=feat_dim
    pe = torch.zeros(L, D, device=device)
    pos = torch.arange(0, L, dtype=torch.float, device=device).unsqueeze(1) # [L, 1]
    div = torch.exp(torch.arange(0, D, 2, device=device) * (-math.log(10000.0) / D)) # [D/2]
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class MusicHighlighter(nn.Module):
    def __init__(self, mel_frames=CHUNK_FRAMES, mel_bands=N_MELS, feat_dim_base=64):
        super().__init__()
        self.mel_frames = mel_frames
        self.mel_bands = mel_bands
        self.feat_dim = feat_dim_base * 4 # Final feature dimension per chunk

        # Convolutional blocks to process each (CHUNK_FRAMES, N_MELS) chunk
        # Input to conv: (B*L_max, 1, CHUNK_FRAMES, N_MELS)
        # Kernel sizes are (time_kernel, freq_kernel)
        def conv_block(ic, oc, k_t, k_f, s_t, s_f):
            return nn.Sequential(
                nn.Conv2d(ic, oc, kernel_size=(k_t, k_f), stride=(s_t, s_f)),
                nn.BatchNorm2d(oc),
                nn.ReLU()
            )
        # Example: Process N_MELS down to 1, and CHUNK_FRAMES down significantly
        self.conv_blocks = nn.Sequential(
            conv_block(1,           feat_dim_base,     k_t=4, k_f=self.mel_bands, s_t=2, s_f=1), # Output freq_dim=1
            conv_block(feat_dim_base, feat_dim_base*2, k_t=4, k_f=1,            s_t=2, s_f=1),
            conv_block(feat_dim_base*2, self.feat_dim,    k_t=4, k_f=1,            s_t=2, s_f=1)
        )
        # After conv_blocks and maxpool, each chunk is (self.feat_dim)

        # Attention mechanism and regression head
        self.attn_mlp = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(self.feat_dim, self.feat_dim), nn.Tanh(), nn.Dropout(0.5),
            nn.Linear(self.feat_dim, 1)
        )
        self.regr_head = nn.Sequential(
            nn.Linear(self.feat_dim, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, TARGET_DIM),    # TARGET_DIM is 2 (start, end)
            nn.Sigmoid()                   # Output normalized 0-1
        )

    def forward(self, x_batch_padded, chunk_lengths):
        # x_batch_padded shape: (B, L_max, N_MELS, CHUNK_FRAMES) <- Note: Mels and Chunks swapped from your original
        # Transpose input to be (B, L_max, CHUNK_FRAMES, N_MELS) for Conv2D
        x_batch_padded = x_batch_padded.permute(0, 1, 3, 2) 
        
        B, L_max, T_chunk, M_bands = x_batch_padded.shape # B, L_max, CHUNK_FRAMES, N_MELS
        dev = x_batch_padded.device

        # Reshape for convolutional blocks: (B * L_max, 1, CHUNK_FRAMES, N_MELS)
        x_reshaped = x_batch_padded.reshape(B * L_max, 1, T_chunk, M_bands)
        
        # Pass through convolutional blocks
        # h_conv shape: (B * L_max, self.feat_dim, H_out, W_out)
        h_conv = self.conv_blocks(x_reshaped)
        
        # Global Max Pooling over the (now small) spatial dimensions from conv
        # Squeeze out W_out if it's 1 (due to k_f=N_MELS), then max over H_out
        if h_conv.size(3) == 1: # If frequency dimension was collapsed
            h_conv = h_conv.squeeze(3) # (B*L_max, self.feat_dim, H_out)
        h_chunk_features = torch.max(h_conv, dim=2)[0] # (B*L_max, self.feat_dim)
        
        # Reshape back to (B, L_max, self.feat_dim)
        h_sequence = h_chunk_features.view(B, L_max, self.feat_dim)

        # Add positional encoding
        pe = positional_encoding(L_max, self.feat_dim, dev).unsqueeze(0) # (1, L_max, self.feat_dim)
        h_sequence_pe = h_sequence + pe # Broadcast PE

        # Attention mechanism
        # Flatten for MLP: (B * L_max, self.feat_dim)
        attn_input = h_sequence_pe.reshape(B * L_max, self.feat_dim)
        attn_logits = self.attn_mlp(attn_input).view(B, L_max) # (B, L_max)

        # Masking for padded sequences
        mask = torch.arange(L_max, device=dev)[None, :] >= chunk_lengths[:, None] # (B, L_max)
        attn_logits.masked_fill_(mask, -float('inf'))
        
        alpha_t = torch.softmax(attn_logits, dim=1).unsqueeze(1) # (B, 1, L_max) attention weights

        # Weighted sum of chunk features
        # h_sequence is (B, L_max, self.feat_dim)
        weighted_features = torch.bmm(alpha_t, h_sequence).squeeze(1) # (B, self.feat_dim)
        
        # Regression head for start/end times
        output_normalized = self.regr_head(weighted_features) # (B, TARGET_DIM)
        
        return output_normalized, alpha_t.squeeze(1) # Return predictions and attention weights


def run_training(args):
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    try:
        load_chorus_annotations_globally(args.csv)
        split_data = get_or_create_data_split(args.split, args.mel_dir, args.force_recreate_split, args.train_ratio, args.val_ratio, args.test_ratio)
    except Exception as e:
        print(f"FATAL error during data setup: {e}")
        import traceback
        traceback.print_exc()
        return

    train_song_dirs = split_data.get("train", [])
    val_song_dirs = split_data.get("val", [])

    if not train_song_dirs:
        print("No training data found in the split. Aborting training.")
        return

    train_ds = SongDataset(train_song_dirs)
    val_ds = SongDataset(val_song_dirs) if val_song_dirs else None

    if len(train_ds) == 0:
        print("Training dataset is empty after filtering. Aborting training.")
        return
    
    print(f"Using {len(train_ds)} training samples, {len(val_ds) if val_ds else 0} validation samples.")

    ds_args = {
        'batch_size': args.batch,
        'num_workers': args.workers,
        'pin_memory': True if device != 'cpu' else False,
        'collate_fn': collate_fn,
        'persistent_workers': True if args.workers > 0 and device != 'cpu' else False
    }
    
    train_ld = DataLoader(train_ds, shuffle=True, **ds_args)
    val_ld = DataLoader(val_ds, shuffle=False, **ds_args) if val_ds and len(val_ds) > 0 else None

    model = MusicHighlighter(mel_frames=CHUNK_FRAMES, mel_bands=N_MELS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Using Mean Squared Error loss, element-wise to find the minimum later
    crit = nn.MSELoss(reduction='none') 
    best_val_loss = float("inf")

    print(f"Starting training on {device} for {args.epochs} epochs...")
    for ep in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n_items_processed_for_loss = 0 # Count items contributing to loss

        for mel_padded, y_labels_list, chunk_lengths, _ in tqdm(train_ld, desc=f"Epoch {ep}/{args.epochs} [Train]"):
            if mel_padded is None or not y_labels_list: # Batch might be empty if all items failed
                continue
                
            mel_padded = mel_padded.to(device)
            chunk_lengths = chunk_lengths.to(device)
            # y_labels_list elements (tensors) will be moved to device inside the item loop
            
            opt.zero_grad()
            # pred_batch shape: [B, TARGET_DIM] (e.g., [B, 2] for start, end)
            pred_batch, _ = model(mel_padded, chunk_lengths)
            
            current_batch_min_mse_losses = []
            for i in range(pred_batch.size(0)): # Iterate over each item in the batch
                pred_single_item = pred_batch[i] # Shape: [TARGET_DIM]
                true_targets_for_item = y_labels_list[i].to(device) # Shape: [N_choruses_for_this_item, TARGET_DIM]

                if true_targets_for_item.nelement() == 0: # No ground truth choruses for this item
                    continue 

                # Expand prediction to compare with all true targets for this item
                # pred_single_item: [TARGET_DIM] -> [1, TARGET_DIM] -> [N_choruses, TARGET_DIM]
                pred_expanded = pred_single_item.unsqueeze(0).expand_as(true_targets_for_item)
                
                # Calculate MSE for each true target against the single prediction
                # mse_per_target will have shape [N_choruses]
                mse_per_target = crit(pred_expanded, true_targets_for_item).mean(dim=1) # Mean over start/end dimensions
                min_mse_for_item = torch.min(mse_per_target) # Find the MSE of the "closest" ground truth
                current_batch_min_mse_losses.append(min_mse_for_item)
            
            if not current_batch_min_mse_losses: # If all items in batch had no targets or were skipped
                continue

            # Average the minimum MSEs across the items in the batch
            batch_loss = torch.stack(current_batch_min_mse_losses).mean()
            
            batch_loss.backward()
            opt.step()
            
            train_loss_sum += batch_loss.item() * len(current_batch_min_mse_losses) # Accumulate loss weighted by number of items
            train_n_items_processed_for_loss += len(current_batch_min_mse_losses)

        avg_train_loss = train_loss_sum / train_n_items_processed_for_loss if train_n_items_processed_for_loss > 0 else 0.0

        # Validation phase
        avg_val_loss_str = "N/A"
        avg_val_mae_sec_str = "N/A"
        current_epoch_val_loss_aggregated = float('inf')

        if val_ld:
            model.eval()
            val_loss_sum = 0.0
            val_n_items_processed_for_loss = 0
            total_val_mae_sec_sum = 0.0

            with torch.no_grad():
                for mel_padded, y_labels_list, chunk_lengths, y_durations_batch in tqdm(val_ld, desc=f"Epoch {ep}/{args.epochs} [Val]  "):
                    if mel_padded is None or not y_labels_list:
                        continue
                    
                    mel_padded = mel_padded.to(device)
                    chunk_lengths = chunk_lengths.to(device)
                    y_durations_batch = y_durations_batch.to(device) # Song durations for MAE calculation

                    pred_batch, _ = model(mel_padded, chunk_lengths)
                    
                    current_batch_val_min_mse_losses = []
                    current_batch_val_min_mae_sec = []

                    for i in range(pred_batch.size(0)):
                        pred_single_item_norm = pred_batch[i] # Normalized prediction
                        true_targets_for_item_norm = y_labels_list[i].to(device)
                        item_duration_sec = y_durations_batch[i]

                        if true_targets_for_item_norm.nelement() == 0:
                            continue

                        # MSE Loss (on normalized values)
                        pred_expanded_norm = pred_single_item_norm.unsqueeze(0).expand_as(true_targets_for_item_norm)
                        mse_per_target_norm = crit(pred_expanded_norm, true_targets_for_item_norm).mean(dim=1)
                        min_mse_for_item_norm = torch.min(mse_per_target_norm)
                        current_batch_val_min_mse_losses.append(min_mse_for_item_norm)

                        # MAE in Seconds Calculation (compare to closest target)
                        pred_sec_single = pred_single_item_norm * item_duration_sec # Denormalize prediction
                        true_sec_targets = true_targets_for_item_norm * item_duration_sec # Denormalize all GTs
                        
                        pred_sec_expanded = pred_sec_single.unsqueeze(0).expand_as(true_sec_targets)
                        
                        # MAE for each target pair (start, end), then average for the pair
                        mae_per_target_pair_sec = torch.abs(pred_sec_expanded - true_sec_targets).mean(dim=1) # Shape: [N_choruses_i]
                        min_mae_sec_for_item = torch.min(mae_per_target_pair_sec)
                        current_batch_val_min_mae_sec.append(min_mae_sec_for_item)
                    
                    if current_batch_val_min_mse_losses:
                        val_loss_sum += torch.stack(current_batch_val_min_mse_losses).sum().item() # Sum of minimums
                        val_n_items_processed_for_loss += len(current_batch_val_min_mse_losses)
                    if current_batch_val_min_mae_sec:
                        total_val_mae_sec_sum += torch.stack(current_batch_val_min_mae_sec).sum().item()
            
            if val_n_items_processed_for_loss > 0:
                current_epoch_val_loss_aggregated = val_loss_sum / val_n_items_processed_for_loss
                avg_val_loss_str = f"{current_epoch_val_loss_aggregated:.4f}"
                avg_val_mae_sec = total_val_mae_sec_sum / val_n_items_processed_for_loss # Avg MAE per item
                avg_val_mae_sec_str = f"{avg_val_mae_sec:.2f}s"
        
        print(f"E {ep:02d}/{args.epochs} | Tr L: {avg_train_loss:.4f} | Vl L: {avg_val_loss_str} | Vl MAE: {avg_val_mae_sec_str}", end="")
        
        if val_ld and current_epoch_val_loss_aggregated < best_val_loss:
            best_val_loss = current_epoch_val_loss_aggregated
            try:
                torch.save(model.state_dict(), args.out)
                print(f" -> Saved model to {args.out}")
            except Exception as e:
                print(f" -> ERROR saving model to {args.out}: {e}")
        else:
            print() # Newline if model not saved
            
    best_val_loss_str = f"{best_val_loss:.4f}" if best_val_loss != float('inf') else "N/A"
    print(f"\nTraining done. Best Validation Loss: {best_val_loss_str}")


# --- NEW FUNCTION for Plotting (copied from previous response) ---
def plot_chorus_comparison(salami_id, song_duration_sec, ground_truth_segments_sec, predicted_segment_sec, output_plot_path):
    plt.figure(figsize=(12, 3))
    plt.title(f"Chorus Segments for Salami ID: {salami_id} (Duration: {song_duration_sec:.2f}s)")
    for i, (gt_start, gt_end) in enumerate(ground_truth_segments_sec):
        plt.hlines(y=1, xmin=gt_start, xmax=gt_end, colors='blue', lw=10, label="Ground Truth" if i == 0 else "")
    if predicted_segment_sec is not None:
        pred_start, pred_end = predicted_segment_sec
        plt.hlines(y=0.5, xmin=pred_start, xmax=pred_end, colors='red', lw=10, label="Prediction")
    plt.yticks([0.5, 1], ['Prediction', 'Ground Truth'])
    plt.xlabel("Time (seconds)")
    plt.xlim(0, song_duration_sec)
    plt.ylim(0, 1.5); plt.legend(loc='upper right'); plt.grid(True, axis='x'); plt.tight_layout()
    try:
        os.makedirs(os.path.dirname(output_plot_path), exist_ok=True) # Ensure dir exists
        plt.savefig(output_plot_path)
    except Exception as e: print(f"Error saving plot {output_plot_path}: {e}")
    plt.close()


# --- NEW FUNCTION for Evaluation/Inference (copied from previous response, with minor robustification) ---
def run_evaluation(args, model_path, eval_song_dirs, eval_output_base_dir):
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"\n--- Running Evaluation on {len(eval_song_dirs)} songs ---")
    print(f"Using model: {model_path}")
    print(f"Outputting results to: {eval_output_base_dir}")
    os.makedirs(eval_output_base_dir, exist_ok=True)

    model = MusicHighlighter(mel_frames=CHUNK_FRAMES, mel_bands=N_MELS).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}. Cannot run evaluation.")
        return
    except Exception as e:
        print(f"ERROR: Could not load model from {model_path}: {e}")
        return
    model.eval()

    eval_ds = SongDataset(eval_song_dirs)
    if not eval_ds or len(eval_ds) == 0:
        print("No valid samples found for evaluation in the specified split.")
        return
        
    ds_args = {'batch_size': 1, 'num_workers': args.workers, 'pin_memory': True if device != 'cpu' else False, 
               'collate_fn': collate_fn, 'persistent_workers': True if args.workers > 0 and device != 'cpu' else False}
    eval_ld = DataLoader(eval_ds, shuffle=False, **ds_args)
    all_results = []

    with torch.no_grad():
        for batch_idx, data_tuple in enumerate(tqdm(eval_ld, desc="Evaluating songs")):
            if data_tuple[0] is None: # Check if collate_fn returned None for mels
                # Try to get salami_id from the original eval_song_dirs list if possible
                salami_id_for_skip_log = "unknown"
                if batch_idx < len(eval_song_dirs):
                    salami_id_for_skip_log = os.path.basename(eval_song_dirs[batch_idx])
                print(f"Skipping item (Salami ID approx. {salami_id_for_skip_log}) due to invalid data from SongDataset/collate_fn.")
                continue

            mel_padded, y_labels_list, chunk_lengths, y_durations_batch = data_tuple
            mel_padded, chunk_lengths = mel_padded.to(device), chunk_lengths.to(device)
            
            pred_norm, _ = model(mel_padded, chunk_lengths) 
            pred_norm_single = pred_norm[0].cpu().numpy()
            
            true_labels_norm_list_of_arrays = [arr.cpu().numpy() for arr in y_labels_list[0]]
            song_duration_sec = y_durations_batch[0].item()
            
            salami_id = "unknown" # Default if we can't get it
            # This relies on eval_ds.items being in the same order and length as what DataLoader iterates
            # which should be true for shuffle=False and batch_size=1 if no items are skipped by __getitem__ / collate_fn
            if batch_idx < len(eval_ds.items):
                 _, original_song_dir_path_for_item = eval_ds.items[batch_idx]
                 salami_id = os.path.basename(original_song_dir_path_for_item)
            else: # Fallback if something went wrong with indexing
                 if batch_idx < len(eval_song_dirs): # Try from the input list to eval_ds
                     salami_id = os.path.basename(eval_song_dirs[batch_idx])
                 else:
                     salami_id = f"unknown_song_eval_idx_{batch_idx}"


            pred_sec = pred_norm_single * song_duration_sec
            true_labels_sec_list = [label_norm * song_duration_sec for label_norm in true_labels_norm_list_of_arrays]
            song_output_dir = os.path.join(eval_output_base_dir, salami_id)
            os.makedirs(song_output_dir, exist_ok=True)

            output_data = {
                "salami_id": salami_id, "song_duration_sec": song_duration_sec,
                "predicted_chorus_sec": [float(pred_sec[0]), float(pred_sec[1])],
                "ground_truth_choruses_sec": [arr.tolist() for arr in true_labels_sec_list],
                "predicted_chorus_norm": [float(pred_norm_single[0]), float(pred_norm_single[1])],
                "ground_truth_choruses_norm": [arr.tolist() for arr in true_labels_norm_list_of_arrays]
            }
            with open(os.path.join(song_output_dir, "prediction_vs_truth.json"), 'w') as f:
                json.dump(output_data, f, indent=4)
            
            plot_chorus_comparison(salami_id, song_duration_sec, true_labels_sec_list,
                                   pred_sec, os.path.join(song_output_dir, "comparison_plot.png"))
            all_results.append(output_data)

    print(f"--- Evaluation Complete ---")
    print(f"Saved {len(all_results)} evaluation results to subdirectories in {eval_output_base_dir}")


# --- MODIFIED main() function (copied from previous response) ---
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train or Evaluate MusicHighlighter for SALAMI Chorus")
    p.add_argument("--csv", default=DEFAULT_AUGMENTED_CHORUS_CSV, help="Path to chorus CSV")
    p.add_argument("--mel_dir", default=DEFAULT_PRECOMPUTED_MEL_BASE_DIR, help="Base dir of Mel spectrograms")
    p.add_argument("--split", default=DEFAULT_SPLIT_JSON_PATH, help="Path to data split JSON")
    p.add_argument("--force_recreate_split", action="store_true", help="Recreate split JSON")
    p.add_argument("--epochs", type=int, default=50, help="Epochs for training")
    p.add_argument("--batch", type=int, default=8, help="Batch size for training")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    p.add_argument("--out", default="salami_chorus_highlighter_model.pt", help="Path to save/load model")
    p.add_argument("--train_ratio", type=float, default=0.7, help="Train split ratio")
    p.add_argument("--val_ratio", type=float, default=0.15, help="Validation split ratio")
    p.add_argument("--test_ratio", type=float, default=0.15, help="Test split ratio")
    p.add_argument("--evaluate", action="store_true", help="Run in evaluation mode instead of training.")
    p.add_argument("--eval_split", type=str, default="test", choices=["train", "val", "test"], help="Which data split to use for evaluation.")
    p.add_argument("--eval_output_dir", type=str, default="evaluation_outputs", help="Directory to save evaluation results.")
    args = p.parse_args()

    if not (0 < args.train_ratio <= 1 and 0 <= args.val_ratio < 1 and 0 <= args.test_ratio < 1 and \
            math.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0)):
        if not args.evaluate: 
            p.error("Split ratios must be valid (0-1) and sum to 1.0 for training.")

    os.makedirs(args.mel_dir, exist_ok=True)
    print(f"INFO: Mels from: {args.mel_dir}")
    print(f"INFO: Annotations from: {args.csv}")
    print(f"INFO: Data split JSON: {args.split}")

    if args.evaluate:
        if not os.path.exists(args.out):
            print(f"ERROR: Model file for evaluation not found at {args.out}. Train a model first or provide correct path.")
            exit()
        try:
            load_chorus_annotations_globally(args.csv)
            split_data = get_or_create_data_split(args.split, args.mel_dir, False) 
            eval_song_dirs_for_mode = split_data.get(args.eval_split, [])
            if not eval_song_dirs_for_mode:
                print(f"ERROR: No songs found in the '{args.eval_split}' split of {args.split}. Cannot evaluate.")
                exit()
            run_evaluation(args, args.out, eval_song_dirs_for_mode, args.eval_output_dir)
        except Exception as e:
            print(f"FATAL error during evaluation setup or execution: {e}")
            import traceback; traceback.print_exc()
    else: 
        if args.force_recreate_split: print("INFO: Will force recreate split file.")
        run_training(args)