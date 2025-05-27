import os
import json
import random
import librosa
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTModel

SPLIT_FILE = 'dataset_split.json'

# Split handling

def save_split(split_dict, path=SPLIT_FILE):
    with open(path, 'w') as f:
        json.dump(split_dict, f, indent=2)

def load_split(path=SPLIT_FILE):
    with open(path, 'r') as f:
        return json.load(f)

# Dataset class

class HighlightDataset(Dataset):
    def __init__(self, root_folder, transform=None, visualize=False, song_list=None):
        self.root_folder = root_folder
        self.transform = transform
        self.visualize = visualize
        self.samples = []
        self._load_metadata(song_list)

    def _load_metadata(self, song_list=None):
        for song_folder in os.listdir(self.root_folder):
            if song_list and song_folder not in song_list:
                continue

            song_path = os.path.join(self.root_folder, song_folder)
            metadata_file = os.path.join(song_path, 'metadata.txt')
            if os.path.isfile(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = f.readlines()
                    meta_dict = {}
                    for line in metadata:
                        key, val = line.strip().split(': ', 1)
                        meta_dict[key] = val

                    timestamp = float(meta_dict['Detected Timestamp'].split()[0])
                    genre = meta_dict.get('Genres', 'Unknown')
                    duration = float(meta_dict['Full Song Duration'].split()[0])
                    audio_path = os.path.join(song_path, 'full_song.mp3')

                    if os.path.isfile(audio_path):
                        self.samples.append((audio_path, timestamp, genre, duration))
                except Exception as e:
                    print(f"❌ Failed to process metadata in {song_folder}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, timestamp, genre, duration = self.samples[idx]
        audio, sr = librosa.load(audio_path, sr=22050, mono=True)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        if self.visualize:
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Mel-Spectrogram: {os.path.basename(audio_path)}')
            plt.tight_layout()
            plt.savefig(f'mel_spectrogram_{idx}.png')
            plt.close()

        mel_spec_db = torch.tensor(mel_spec_db).unsqueeze(0)

        if self.transform:
            mel_spec_db = self.transform(mel_spec_db)

        timestamp_tensor = torch.tensor([timestamp], dtype=torch.float32)

        return mel_spec_db, timestamp_tensor, genre, duration

# ViT model

class ViTRegression(nn.Module):
    def __init__(self):
        super(ViTRegression, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.regressor = nn.Linear(self.vit.config.hidden_size, 1)

    def forward(self, x):
        x = self.vit(pixel_values=x).pooler_output
        x = self.regressor(x)
        return x

# Training pipeline

def train_model(dataset_path, epochs=100, batch_size=32, lr=1e-5):
    transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.nn.functional.interpolate(x.unsqueeze(0), size=(224, 224)).squeeze(0)),
        transforms.Lambda(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    ])

    all_song_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

    if os.path.exists(SPLIT_FILE):
        print("🔄 Loading dataset split from file...")
        split = load_split()
    else:
        print("🆕 Creating new dataset split...")
        random.shuffle(all_song_folders)
        num_total = len(all_song_folders)
        train_end = int(0.7 * num_total)
        val_end = int(0.85 * num_total)

        split = {
            'train': all_song_folders[:train_end],
            'val': all_song_folders[train_end:val_end],
            'test': all_song_folders[val_end:]
        }
        save_split(split)

    train_dataset = HighlightDataset(dataset_path, transform=transform, song_list=split['train'], visualize=False)
    val_dataset = HighlightDataset(dataset_path, transform=transform, song_list=split['val'], visualize=False)
    test_dataset = HighlightDataset(dataset_path, transform=transform, song_list=split['test'], visualize=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViTRegression().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets, _, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            if inputs.shape[1] != 3:
                inputs = inputs.repeat(1, 3, 1, 1)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets.squeeze())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

    torch.save(model.state_dict(), 'highlight_detector_vit.pth')

    # Plot training loss
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.savefig('training_loss.png')

    # Testing and saving results
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for i, (inputs, targets, genre, duration) in enumerate(test_loader):
            inputs = inputs.to(device)
            if inputs.shape[1] != 3:
                inputs = inputs.repeat(1, 3, 1, 1)

            output = model(inputs).item()
            predictions.append(output)
            actuals.append(targets.item())
            print(f'Predicted: {output:.2f} sec, Actual: {targets.item()} sec, Genre: {genre[0]}, Duration: {duration.item()} sec')

    # Plot predictions vs actual
    plt.figure()
    plt.scatter(actuals, predictions, alpha=0.7)
    plt.xlabel('Actual Timestamp (sec)')
    plt.ylabel('Predicted Timestamp (sec)')
    plt.title('Actual vs Predicted Timestamps')
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
    plt.savefig('actual_vs_predicted.png')

# Usage
if __name__ == "__main__":
    dataset_folder = 'Song_data(CopyrightFree)'
    train_model(dataset_folder)