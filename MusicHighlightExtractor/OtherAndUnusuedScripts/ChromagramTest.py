import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os


def frame_audio(y, sr, frame_duration=5.0):
    frame_length = int(frame_duration * sr)
    num_frames = len(y) // frame_length
    return [y[i * frame_length:(i + 1) * frame_length] for i in range(num_frames)]


def extract_features_per_frame(frames, sr, n_mfcc=13):
    mfccs_list = []
    chroma_list = []

    for frame in frames:
        mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=n_mfcc)
        chroma = librosa.feature.chroma_stft(y=frame, sr=sr)
        mfccs_list.append(mfcc)
        chroma_list.append(chroma)

    return mfccs_list, chroma_list


def plot_frame_features(mfccs_list, chroma_list, sr, frame_duration):
    for i, (mfcc, chroma) in enumerate(zip(mfccs_list, chroma_list)):
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        librosa.display.specshow(mfcc, x_axis='time', sr=sr)
        plt.colorbar()
        plt.title(f'MFCCs (Frame {i + 1}, {i * frame_duration:.1f}-{(i + 1) * frame_duration:.1f} sec)')

        plt.subplot(2, 1, 2)
        librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', sr=sr)
        plt.colorbar()
        plt.title(f'Chromagram (Frame {i + 1}, {i * frame_duration:.1f}-{(i + 1) * frame_duration:.1f} sec)')

        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    path = 'Song_data(CopyrightFree)/Alan_Walker_-_Dreamer/full_song.mp3'
    y, sr = librosa.load(path, sr=22050)
    frames = frame_audio(y, sr, frame_duration=5.0)  # 5 sec frames
    mfccs, chromas = extract_features_per_frame(frames, sr)
    plot_frame_features(mfccs, chromas, sr, frame_duration=5.0)
