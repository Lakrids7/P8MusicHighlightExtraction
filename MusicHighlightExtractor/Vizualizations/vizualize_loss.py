import re
import matplotlib.pyplot as plt

# Paste your training log data here
log_data = """
Using 3160 train, 677 val samples.
Starting training on cuda...
E 01/100 | Tr Loss: 0.0200 | Vl Loss: 0.0181 -> Saved highlighter_regr_precomp_short.pt
E 02/100 | Tr Loss: 0.0187 | Vl Loss: 0.0195
E 03/100 | Tr Loss: 0.0185 | Vl Loss: 0.0181
E 04/100 | Tr Loss: 0.0186 | Vl Loss: 0.0180 -> Saved highlighter_regr_precomp_short.pt
E 05/100 | Tr Loss: 0.0182 | Vl Loss: 0.0178 -> Saved highlighter_regr_precomp_short.pt
E 06/100 | Tr Loss: 0.0180 | Vl Loss: 0.0178 -> Saved highlighter_regr_precomp_short.pt
E 07/100 | Tr Loss: 0.0180 | Vl Loss: 0.0179
E 08/100 | Tr Loss: 0.0180 | Vl Loss: 0.0179
E 09/100 | Tr Loss: 0.0177 | Vl Loss: 0.0179
E 10/100 | Tr Loss: 0.0178 | Vl Loss: 0.0176 -> Saved highlighter_regr_precomp_short.pt
E 11/100 | Tr Loss: 0.0177 | Vl Loss: 0.0177
E 12/100 | Tr Loss: 0.0175 | Vl Loss: 0.0177
E 13/100 | Tr Loss: 0.0176 | Vl Loss: 0.0178
E 14/100 | Tr Loss: 0.0174 | Vl Loss: 0.0174 -> Saved highlighter_regr_precomp_short.pt
E 15/100 | Tr Loss: 0.0174 | Vl Loss: 0.0176
E 16/100 | Tr Loss: 0.0171 | Vl Loss: 0.0179
E 17/100 | Tr Loss: 0.0172 | Vl Loss: 0.0177
E 18/100 | Tr Loss: 0.0171 | Vl Loss: 0.0180
E 19/100 | Tr Loss: 0.0171 | Vl Loss: 0.0179
E 20/100 | Tr Loss: 0.0167 | Vl Loss: 0.0177
E 21/100 | Tr Loss: 0.0169 | Vl Loss: 0.0186
E 22/100 | Tr Loss: 0.0167 | Vl Loss: 0.0185
E 23/100 | Tr Loss: 0.0166 | Vl Loss: 0.0177
E 24/100 | Tr Loss: 0.0165 | Vl Loss: 0.0177
E 25/100 | Tr Loss: 0.0161 | Vl Loss: 0.0208
E 26/100 | Tr Loss: 0.0162 | Vl Loss: 0.0177
E 27/100 | Tr Loss: 0.0159 | Vl Loss: 0.0182
E 28/100 | Tr Loss: 0.0157 | Vl Loss: 0.0186
E 29/100 | Tr Loss: 0.0155 | Vl Loss: 0.0183
E 30/100 | Tr Loss: 0.0153 | Vl Loss: 0.0187
E 31/100 | Tr Loss: 0.0148 | Vl Loss: 0.0187
E 32/100 | Tr Loss: 0.0149 | Vl Loss: 0.0205
E 33/100 | Tr Loss: 0.0145 | Vl Loss: 0.0204
E 34/100 | Tr Loss: 0.0139 | Vl Loss: 0.0197
E 35/100 | Tr Loss: 0.0131 | Vl Loss: 0.0197
E 36/100 | Tr Loss: 0.0127 | Vl Loss: 0.0200
E 37/100 | Tr Loss: 0.0120 | Vl Loss: 0.0202
E 38/100 | Tr Loss: 0.0118 | Vl Loss: 0.0214
E 39/100 | Tr Loss: 0.0111 | Vl Loss: 0.0216
E 40/100 | Tr Loss: 0.0103 | Vl Loss: 0.0205
E 41/100 | Tr Loss: 0.0095 | Vl Loss: 0.0207
E 42/100 | Tr Loss: 0.0086 | Vl Loss: 0.0208
E 43/100 | Tr Loss: 0.0082 | Vl Loss: 0.0203
E 44/100 | Tr Loss: 0.0076 | Vl Loss: 0.0239
E 45/100 | Tr Loss: 0.0074 | Vl Loss: 0.0207
E 46/100 | Tr Loss: 0.0065 | Vl Loss: 0.0219
E 47/100 | Tr Loss: 0.0063 | Vl Loss: 0.0227
E 48/100 | Tr Loss: 0.0056 | Vl Loss: 0.0225
E 49/100 | Tr Loss: 0.0059 | Vl Loss: 0.0211
E 50/100 | Tr Loss: 0.0051 | Vl Loss: 0.0232
E 51/100 | Tr Loss: 0.0047 | Vl Loss: 0.0233
E 52/100 | Tr Loss: 0.0045 | Vl Loss: 0.0205
E 53/100 | Tr Loss: 0.0044 | Vl Loss: 0.0222
E 54/100 | Tr Loss: 0.0043 | Vl Loss: 0.0221
E 55/100 | Tr Loss: 0.0040 | Vl Loss: 0.0223
E 56/100 | Tr Loss: 0.0038 | Vl Loss: 0.0221
E 57/100 | Tr Loss: 0.0036 | Vl Loss: 0.0235
E 58/100 | Tr Loss: 0.0037 | Vl Loss: 0.0215
E 59/100 | Tr Loss: 0.0035 | Vl Loss: 0.0218
E 60/100 | Tr Loss: 0.0034 | Vl Loss: 0.0218
E 61/100 | Tr Loss: 0.0034 | Vl Loss: 0.0216
E 62/100 | Tr Loss: 0.0033 | Vl Loss: 0.0219
E 63/100 | Tr Loss: 0.0033 | Vl Loss: 0.0215
E 64/100 | Tr Loss: 0.0033 | Vl Loss: 0.0221
E 65/100 | Tr Loss: 0.0032 | Vl Loss: 0.0219
E 66/100 | Tr Loss: 0.0030 | Vl Loss: 0.0218
E 67/100 | Tr Loss: 0.0032 | Vl Loss: 0.0213
E 68/100 | Tr Loss: 0.0031 | Vl Loss: 0.0208
E 69/100 | Tr Loss: 0.0030 | Vl Loss: 0.0217
E 70/100 | Tr Loss: 0.0025 | Vl Loss: 0.0214
E 71/100 | Tr Loss: 0.0027 | Vl Loss: 0.0211
E 72/100 | Tr Loss: 0.0026 | Vl Loss: 0.0218
E 73/100 | Tr Loss: 0.0026 | Vl Loss: 0.0209
E 74/100 | Tr Loss: 0.0025 | Vl Loss: 0.0214
E 75/100 | Tr Loss: 0.0027 | Vl Loss: 0.0209
E 76/100 | Tr Loss: 0.0028 | Vl Loss: 0.0227
E 77/100 | Tr Loss: 0.0028 | Vl Loss: 0.0210
E 78/100 | Tr Loss: 0.0027 | Vl Loss: 0.0221
E 79/100 | Tr Loss: 0.0026 | Vl Loss: 0.0203
E 80/100 | Tr Loss: 0.0025 | Vl Loss: 0.0220
E 81/100 | Tr Loss: 0.0023 | Vl Loss: 0.0219
E 82/100 | Tr Loss: 0.0026 | Vl Loss: 0.0220
E 83/100 | Tr Loss: 0.0024 | Vl Loss: 0.0220
E 84/100 | Tr Loss: 0.0023 | Vl Loss: 0.0210
E 85/100 | Tr Loss: 0.0023 | Vl Loss: 0.0211
E 86/100 | Tr Loss: 0.0023 | Vl Loss: 0.0213
E 87/100 | Tr Loss: 0.0024 | Vl Loss: 0.0206
E 88/100 | Tr Loss: 0.0024 | Vl Loss: 0.0204
E 89/100 | Tr Loss: 0.0022 | Vl Loss: 0.0201
E 90/100 | Tr Loss: 0.0021 | Vl Loss: 0.0223
E 91/100 | Tr Loss: 0.0025 | Vl Loss: 0.0209
E 92/100 | Tr Loss: 0.0021 | Vl Loss: 0.0217
E 93/100 | Tr Loss: 0.0021 | Vl Loss: 0.0200
E 94/100 | Tr Loss: 0.0022 | Vl Loss: 0.0216
E 95/100 | Tr Loss: 0.0024 | Vl Loss: 0.0209
E 96/100 | Tr Loss: 0.0022 | Vl Loss: 0.0211
E 97/100 | Tr Loss: 0.0019 | Vl Loss: 0.0205
E 98/100 | Tr Loss: 0.0020 | Vl Loss: 0.0207
E 99/100 | Tr Loss: 0.0020 | Vl Loss: 0.0206
E 100/100 | Tr Loss: 0.0019 | Vl Loss: 0.0212
Done. Best Val Loss: 0.0174
"""

# Regular expression to extract epoch, train loss, and validation loss
# Updated regex to handle potential spaces around numbers and colon/pipe
regex = re.compile(r"E\s*(\d+)/\d+\s*\|\s*Tr Loss:\s*([\d.]+)\s*\|\s*Vl Loss:\s*([\d.]+)")

epochs = []
train_losses = []
val_losses = []
saved_epochs = []

# Parse the data line by line
for line in log_data.strip().split('\n'):
    match = regex.search(line)
    if match:
        epoch = int(match.group(1))
        tr_loss = float(match.group(2))
        vl_loss = float(match.group(3))

        epochs.append(epoch)
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        if "-> Saved" in line:
            saved_epochs.append(epoch)

# Find the best validation loss and epoch reported at the end
best_val_loss_reported = 0.0174 # From the "Done" line
# Find it programmatically too (assuming it might not be the overall minimum)
best_val_loss = min(val_losses)
best_epoch = epochs[val_losses.index(best_val_loss)]

# --- Plotting ---
plt.figure(figsize=(12, 6))

# Plot losses
plt.plot(epochs, train_losses, label='Training Loss', color='blue', alpha=0.8)
plt.plot(epochs, val_losses, label='Validation Loss', color='orange', alpha=0.8)

# Highlight saved epochs (where validation loss improved *at that point*)
# plt.scatter(saved_epochs, [val_losses[epoch-1] for epoch in saved_epochs],
#             color='red', marker='o', s=50, label='Model Saved', zorder=5) # Uncomment if you want to see all saves

# Highlight the best epoch found
plt.scatter(best_epoch, best_val_loss, color='red', marker='*', s=150, label=f'Best Val Loss ({best_val_loss:.4f}) at Epoch {best_epoch}', zorder=5)
plt.axvline(x=best_epoch, color='red', linestyle='--', linewidth=1, label=f'Best Epoch ({best_epoch})')


# Add labels and title
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss per Epoch")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(bottom=0) # Start y-axis at 0
plt.xlim(left=0)   # Start x-axis at 0

# Improve layout and display
plt.tight_layout()
plt.show()