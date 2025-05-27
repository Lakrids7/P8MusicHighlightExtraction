#Script that attempts to implement a 'traditional' implementation of music highlight extraction/thumbnailing.
#Overall structure is:
    # 1. Divides a song into frames of around 20ms
    # 2. Computes MFCCs and Chromagrams of each frame
    # 3. Computes a self-similarity matrix, comparing these features across the song to find the most repeated segments
    # ?

import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import time
from tqdm import tqdm

def load_audio(file_path, sr=22050):
    audio, sr = librosa.load(file_path, sr=sr)
    return audio, sr

def extract_features(audio, sr, frame_length=0.02):
    hop_length = int(sr * frame_length)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=hop_length)

    # Chromagram
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=hop_length)

    # Concatenate features vertically
    features = np.vstack((mfcc, chroma))

    return features.T, hop_length

def compute_self_similarity(features):
    features_norm = librosa.util.normalize(features, axis=1)
    similarity_matrix = np.dot(features_norm, features_norm.T)
    return similarity_matrix

def find_best_30s_segment(similarity_matrix, sr, hop_length, segment_duration=30, stride=5):
    num_frames = similarity_matrix.shape[0]
    target_frames = int((segment_duration * sr) / hop_length)

    best_score = -np.inf
    best_segment = (0, target_frames)

    print(f"Searching for best {segment_duration}s highlight segment ({target_frames} frames)...")
    t_start = time.time()

    for start in tqdm(range(0, num_frames - target_frames, stride), desc="Searching segments"):
        segment = similarity_matrix[start:start+target_frames, start:start+target_frames]
        score = np.mean(segment)

        if score > best_score:
            best_score = score
            best_segment = (start, start + target_frames)

    t_end = time.time()
    print(f"Search completed in {t_end - t_start:.2f} seconds")
    print(f"Best segment score: {best_score:.4f}")

    start_time = best_segment[0] * hop_length / sr
    end_time = best_segment[1] * hop_length / sr

    return start_time, end_time

def save_thumbnail(audio, sr, start_time, end_time, output_path='highlight_segment.wav'):
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    thumbnail = audio[start_sample:end_sample]
    sf.write(output_path, thumbnail, sr)

def plot_similarity(similarity_matrix):
    plt.figure(figsize=(8, 6))
    librosa.display.specshow(similarity_matrix, x_axis='time', y_axis='time', cmap='coolwarm')
    plt.colorbar()
    plt.title('Self-Similarity Matrix')
    plt.show()

if __name__ == "__main__":
    audio_path = 'Song_data/Ed_Sheeran_-_Thinking_out_Loud/full_song.mp3'

    print("Loading audio...")
    audio, sr = load_audio(audio_path)

    print("Extracting features...")
    features, hop_length = extract_features(audio, sr)

    print("Computing self-similarity matrix...")
    similarity_matrix = compute_self_similarity(features)

    plot_similarity(similarity_matrix)

    start, end = find_best_30s_segment(similarity_matrix, sr, hop_length, segment_duration=30)
    print(f"Highlight Segment: {start:.2f}s to {end:.2f}s")

    save_thumbnail(audio, sr, start, end, 'highlight_segment.wav')
    print("Highlight segment saved as 'highlight_segment.wav'")
