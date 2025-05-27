import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np # Needed for plotting

# Constants required by the model architecture
# Using the constants from the original script for consistency
N_MELS = 128      # Number of mel frequency bands
CHUNK_FRAMES = 128 # Number of frames per audio chunk
TARGET_DIM = 2    # Dimension of the output (e.g., start and end timestamps)

def positional_encoding(L, D, device):
    """
    Generates positional encoding for a sequence of length L.

    Args:
        L (int): Sequence length (number of chunks).
        D (int): Embedding dimension.
        device (torch.device): Device for tensor placement.

    Returns:
        torch.Tensor: Positional encoding tensor (L, D).
    """
    pe = torch.zeros(L, D, device=device)
    pos = torch.arange(0, L, dtype=torch.float, device=device).unsqueeze(1)
    # Calculate the division term using log-space for numerical stability
    div = torch.exp(torch.arange(0, D, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / D))
    pe[:, 0::2] = torch.sin(pos * div) # Apply sin to even indices
    pe[:, 1::2] = torch.cos(pos * div) # Apply cos to odd indices
    return pe

class MusicHighlighter(nn.Module):
    """
    PyTorch model for predicting song highlight timestamps using attention.
    Processes a sequence of audio chunks.
    """
    def __init__(self, dim=64):
        """
        Args:
            dim (int): Base dimension for convolutional layers.
        """
        super().__init__()
        # The feature dimension after convolutions will be dim * 4
        self.feat_dim = dim * 4

        # Helper function for convolutional blocks
        def conv_block(in_channels, out_channels, kernel_size, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        # Convolutional layers to process each audio chunk
        # Input per chunk is treated as (1, CHUNK_FRAMES, N_MELS) -> after permute/reshape (1, T, M)
        # Input to conv_blocks is (B*L, 1, T, M) where T=CHUNK_FRAMES, M=N_MELS
        # Original kernel shape was (3, N_MELS) implying it acts on (time, freq).
        # After transpose in forward pass, input is (B*L, 1, T, M).
        # Kernel (3, N_MELS) on (B*L, 1, T, M) applies over (time, freq)
        self.conv_blocks = nn.Sequential(
            # Input: (B*L, 1, CHUNK_FRAMES, N_MELS)
            conv_block(1, dim, (3, N_MELS), (2, 1)), # Kernel (3, N_MELS) stride (2, 1)
            # Output after 1st conv: (B*L, dim, (CHUNK_FRAMES-3)/2+1, 1) -> let T1=(CHUNK_FRAMES-3)/2+1
            conv_block(dim, dim*2, (4, 1), (2, 1)), # Kernel (4, 1) stride (2, 1)
            # Output after 2nd conv: (B*L, dim*2, (T1-4)/2+1, 1) -> let T2=(T1-4)/2+1
            conv_block(dim*2, self.feat_dim, (4, 1), (2, 1)) # Kernel (4, 1) stride (2, 1)
            # Output after 3rd conv: (B*L, feat_dim, (T2-4)/2+1, 1)
        )
        # After convolutions, the shape is (B*L, feat_dim, final_reduced_T, 1)
        # We then squeeze the last dim and take torch.max over the time dim (dim=2).
        # This gives (B*L, feat_dim)

        # Attention mechanism MLP to compute attention scores per chunk
        self.attn_mlp = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.Tanh(), # Tanh activation
            nn.Dropout(0.5),
            nn.Linear(self.feat_dim, 1) # Output a single attention score (logit) per chunk
        )

        # Regression head to predict start/end times from the weighted features
        self.regr_head = nn.Sequential(
            nn.Linear(self.feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, TARGET_DIM),
            nn.Sigmoid() # Ensure output is between 0 and 1 (relative timestamps)
        )

    def forward(self, x, lengths):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input mel spectrogram chunks.
                             Shape (Batch_size, Max_Seq_Len, CHUNK_FRAMES, N_MELS).
            lengths (torch.Tensor): Lengths of the original sequences before padding.
                                    Shape (Batch_size,).

        Returns:
            tuple: (predictions, attention_weights)
                predictions (torch.Tensor): Predicted timestamps (Batch_size, TARGET_DIM).
                attention_weights (torch.Tensor): Attention weights per chunk (Batch_size, Max_Seq_Len).
        """
        B, L_max, T, M = x.shape # B=Batch, L_max=Max Seq Len, T=CHUNK_FRAMES, M=N_MELS
        dev = x.device

        # Reshape and permute for convolutional layers: (B*L, 1, T, M)
        # Note: The original code permuted (0, 1, 3, 2) and then reshaped.
        # x.permute(0, 1, 3, 2).reshape(B * L_max, 1, M, T) -> (B*L, 1, N_MELS, CHUNK_FRAMES)
        # Let's stick to the original implementation's implicit ordering for conv input.
        # This means the conv kernels (3, N_MELS), (4, 1), (4, 1) operate on
        # the (time, freq) dimensions where time is the 3rd dim (index 2)
        # and freq (N_MELS) is the 4th dim (index 3) of the input (B*L, 1, T, M).
        # So the input to conv is (B*L, 1, CHUNK_FRAMES, N_MELS).
        # The original comment was confusing, let's clarify the convolution behavior.
        # conv(1, dim, (k_t, k_m), (s_t, s_m)) applies k_t over CHUNK_FRAMES (dim 2) and k_m over N_MELS (dim 3).
        # conv(1, dim, (3, N_MELS), (2, 1)): Kernel (3, N_MELS) strides (2, 1). Reduces CHUNK_FRAMES dim, collapses N_MELS dim.
        # Output shape: (B*L, dim, (CHUNK_FRAMES - 3)//2 + 1, (N_MELS - N_MELS)//1 + 1) = (B*L, dim, reduced_T1, 1)
        # conv(dim, dim*2, (4, 1), (2, 1)): Kernel (4, 1) strides (2, 1) on (reduced_T1, 1). Reduces reduced_T1 dim.
        # Output shape: (B*L, dim*2, (reduced_T1 - 4)//2 + 1, 1) = (B*L, dim*2, reduced_T2, 1)
        # conv(dim*2, self.feat_dim, (4, 1), (2, 1)): Kernel (4, 1) strides (2, 1) on (reduced_T2, 1). Reduces reduced_T2 dim.
        # Output shape: (B*L, feat_dim, (reduced_T2 - 4)//2 + 1, 1) = (B*L, feat_dim, final_reduced_T, 1)
        # This matches the code's logic: squeeze(3) removes the last '1' dim, then max(dim=2) takes max over the final_reduced_T dim.

        x = x.view(B * L_max, 1, T, M) # (B*L, 1, CHUNK_FRAMES, N_MELS) - Input to conv blocks

        # Apply convolutional blocks, squeeze last dim, and take max over time dim
        # Output shape after conv_blocks: (B*L, feat_dim, final_reduced_T, 1)
        x = torch.max(self.conv_blocks(x).squeeze(3), dim=2)[0] # (B*L, feat_dim)

        # Reshape back to sequence format: (B, L, feat_dim)
        h_t = x.view(B, L_max, -1)

        # Add positional encoding to chunk features
        pe = positional_encoding(L_max, self.feat_dim, dev).unsqueeze(0) # (1, L, D)
        # Flatten for the attention MLP: (B*L, feat_dim)
        attn_logits = self.attn_mlp((h_t + pe).view(B * L_max, -1)).view(B, L_max) # (B, L)

        # Create a mask for padded chunks based on original lengths
        mask = torch.arange(L_max, device=dev)[None, :] >= lengths[:, None] # (B, L)
        # Apply mask by setting logits of padded elements to -inf
        attn_logits.masked_fill_(mask, -float('inf'))

        # Compute attention weights using softmax
        alpha_t = torch.softmax(attn_logits, dim=1).unsqueeze(1) # (B, 1, L)

        # Compute weighted sum of chunk features using attention weights
        # (B, 1, L) @ (B, L, feat_dim) -> (B, 1, feat_dim) -> squeeze(1) -> (B, feat_dim)
        weighted_features = torch.bmm(alpha_t, h_t).squeeze(1) # (B, feat_dim)

        # Pass weighted features through the regression head to get predictions
        predictions = self.regr_head(weighted_features) # (B, TARGET_DIM)

        # Return predictions and attention weights
        return predictions, alpha_t.squeeze(1)


# --- Plotting Function for Visualization ---

def plot_attention_weights(attention_weights, lengths, title="Attention Weights", max_frames=None):
    """
    Plots attention weights per chunk for a batch, with improved visibility.

    Args:
        attention_weights (torch.Tensor): Tensor of attention weights (Batch_size, Max_Seq_Len).
        lengths (torch.Tensor): Original sequence lengths (Batch_size,).
        title (str): Title for the plot.
        max_frames (int, optional): If provided, limits the x-axis and plotted data
                                    to this number of frames (chunks). Defaults to None.
    """
    attn_np = attention_weights.detach().cpu().numpy()
    lengths_np = lengths.detach().cpu().numpy()
    batch_size, max_seq_len = attn_np.shape

    # Determine the actual length to plot for each sequence
    plot_lengths = np.minimum(lengths_np, max_frames if max_frames is not None else max_seq_len)

    # Determine the x-axis limit
    x_lim = max_frames if max_frames is not None else max_seq_len

    # --- Improved Plotting Settings ---
    plt.figure(figsize=(18, 8)) # Make figure wider and taller
    plt.style.use('seaborn-v0_8-darkgrid') # Use a clear grid style

    for i in range(batch_size):
        # Plot only up to the actual length or max_frames limit
        plot_len_i = plot_lengths[i]
        if plot_len_i > 0:
             # Use a line with markers for better visibility
            plt.plot(np.arange(plot_len_i), attn_np[i, :plot_len_i], marker='o', linestyle='-',
                     linewidth=2.5, markersize=8, label=f'Batch Item {i+1}')

    # --- Apply Larger Font Sizes and Labels ---
    plt.title(title, fontsize=20)
    plt.xlabel("Chunk Index", fontsize=16)
    plt.ylabel("Attention Weight", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Set x-axis limit based on max_frames or max_seq_len
    # Add a small buffer for padding if needed
    plt.xlim(-0.5, x_lim + 0.5) # Start slightly before 0 for clarity

    # Ensure y-axis starts at 0 and goes slightly above max observed attention
    max_attn_val = np.max(attn_np[:, :x_lim]) if x_lim > 0 else 0.1
    plt.ylim(-0.02, max_attn_val * 1.1 + 0.02) # Small buffer below 0 and above max

    plt.legend(fontsize=14) # Larger legend font
    plt.tight_layout() # Adjust layout to prevent overlap
    plt.show()


# --- Example Usage (Demonstration) ---
if __name__ == '__main__':
    # Define constants (already defined above, but good to show here)
    # N_MELS = 128
    # CHUNK_FRAMES = 128
    # TARGET_DIM = 2

    batch_size = 4
    max_seq_len = 100 # Let's use a longer max sequence length for better demo
    feature_dim = 64 # Corresponds to 'dim' in __init__

    # Create dummy data (batch, max_seq_len, CHUNK_FRAMES, N_MELS)
    # Simulate padded sequences with varying lengths
    # Make some sequences shorter than max_seq_len
    dummy_data = torch.randn(batch_size, max_seq_len, CHUNK_FRAMES, N_MELS)
    dummy_lengths = torch.tensor([85, 100, 60, 92], dtype=torch.long) # Original lengths

    # Instantiate the model
    model = MusicHighlighter(dim=feature_dim)

    # Move data to device (CPU in this example)
    device = torch.device("cpu")
    model.to(device)
    dummy_data = dummy_data.to(device)
    dummy_lengths = dummy_lengths.to(device)

    # Perform a forward pass to get attention weights
    # Need to wrap in torch.no_grad() if just doing inference for plotting
    model.eval() # Set model to evaluation mode (disables dropout)
    with torch.no_grad():
        predictions, attention_weights = model(dummy_data, dummy_lengths)

    print("Model Architecture:")
    print(model)

    print("\nDummy Input Shape:", dummy_data.shape)
    print("Dummy Lengths:", dummy_lengths)
    print("\nPredicted Output Shape:", predictions.shape)
    print("Attention Weights Shape:", attention_weights.shape)
    print("\nAttention weights sum for each item (should be ~1):", attention_weights.sum(dim=1))


    # --- Visualize Attention Weights ---

    # Variant 1: Full sequence attention plot
    print("\nPlotting full sequence attention...")
    plot_attention_weights(attention_weights, dummy_lengths, title="Full Sequence Attention Weights")

    # Variant 2: Attention plot truncated at 40 frames (chunks)
    print("Plotting attention truncated at 40 frames...")
    plot_attention_weights(attention_weights, dummy_lengths, title="Attention Weights (First 40 Frames)", max_frames=40)