import subprocess
import re
import os
import requests
import yt_dlp
import librosa
import numpy as np
from scipy.signal import correlate
from pydub import AudioSegment
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import asyncio

# Spotify API Setup
SPOTIFY_CLIENT_ID = ""
SPOTIFY_CLIENT_SECRET = ""

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID,
                                                           client_secret=SPOTIFY_CLIENT_SECRET))

SPOTIFY_TRACK_REGEX = r"https://open\.spotify\.com/track/([a-zA-Z0-9]+)"
SPOTIFY_PLAYLIST_REGEX = r"https://open\.spotify\.com/playlist/([a-zA-Z0-9]+)"
def get_songs_from_playlist(spotify_url):
    """Fetch all track URLs from a Spotify playlist link, handling pagination."""
    match = re.search(SPOTIFY_PLAYLIST_REGEX, spotify_url)
    if not match:
        return None

    playlist_id = match.group(1)
    song_urls = []

    offset = 0
    limit = 100  # Maximum allowed by Spotify per request

    while True:
        results = sp.playlist_items(playlist_id, offset=offset, limit=limit)
        items = results['items']

        if not items:
            break  # No more items left to fetch

        for item in items:
            track = item['track']
            if track:  # Ensure track is not None (Spotify sometimes returns None for unavailable tracks)
                song_url = f"https://open.spotify.com/track/{track['id']}"
                song_urls.append(song_url)

        offset += limit  # Move to the next set of tracks

        # Break loop if all tracks have been fetched
        if len(items) < limit:
            break

    return song_urls

def sanitize_filename(filename):
    """Removes invalid characters from a filename, ensuring Windows compatibility."""
    filename = re.sub(r'[\/:*?"<>|#]', '', filename)  # Remove forbidden characters
    filename = filename.rstrip('.')  # Prevent trailing dots (Windows issue)
    filename = filename.replace(' ', '_')  # Replace spaces with underscores (optional)
    return filename

def create_song_folder(song_name, artist_name):
    """Creates a folder for storing song-related files inside the 'Song_data' directory. Returns None if folder already exists."""
    parent_folder = "Song_data_New"
    os.makedirs(parent_folder, exist_ok=True)

    folder_name = sanitize_filename(f"{artist_name} - {song_name}")
    full_path = os.path.join(parent_folder, folder_name)

    if os.path.exists(full_path):
        print(f"⚠️ Folder already exists for '{artist_name} - {song_name}', skipping.")
        return None  # Signal that processing should be skipped

    os.makedirs(full_path)
    return full_path

async def extract_and_save_preview(preview_start, full_song_file, song_folder):
    """Extracts the matched preview section from the full song and saves it."""
    try:
        if not os.path.exists(full_song_file):
            raise FileNotFoundError(f"File not found: {full_song_file}")

        output_file = os.path.join(song_folder, "extracted_preview.mp3")
        song = AudioSegment.from_file(full_song_file, format="mp3")

        start_ms = preview_start * 1000
        end_ms = start_ms + (30 * 1000)  # Extract 30 seconds
        snippet = song[start_ms:end_ms]
        snippet.export(output_file, format="mp3")
    except Exception as e:
        print(f"❌ Error extracting preview section: {e}")

def get_spotify_preview_url(song_name, artist_name):
    query = f"{song_name} {artist_name}"
    try:
        result = subprocess.run(["node", "preview_finder.js", query], capture_output=True, text=True)
        preview_url = result.stdout.strip()
        print(f"🔗 Spotify preview URL: {preview_url}")
        if "No preview available" in preview_url or result.returncode != 0:
            return None
        return preview_url
    except Exception as e:
        print(f"Error running Node.js script: {e}")
        return None

def download_audio(url, output_filename):
    """Download an audio file from a given URL."""
    response = requests.get(url)
    with open(output_filename, 'wb') as f:
        f.write(response.content)
    return output_filename

def download_full_song(song_name, artist_name, song_folder):
    """Download the full song from YouTube, ensuring accurate matches."""
    full_song_path = os.path.join(song_folder, "full_song.mp3")

    search_terms = [
        f"{song_name} {artist_name} official audio",
        f"{song_name} {artist_name} lyrics",
        f"{song_name} {artist_name} audio only",
        f"{song_name} {artist_name}",
    ]


    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192'
        }],
        'outtmpl': full_song_path[:-4],
        'quiet': True,
        'default_search': 'ytsearch10',
        'noplaylist': True,
        'match_filter': lambda info: None if info.get('duration', 0) <= 300 else 'Video exceeds 5 minutes',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for query in search_terms:
            print(f"🔍 Searching YouTube for: '{query}'")
            try:
                # Extract search results without downloading immediately
                search_results = ydl.extract_info(query, download=False)['entries']

                # Check manually if track_name and artist_name appear in the title
                for video in search_results:
                    title = video_title = video_uploader = ""
                    title = search_results[search_results.index(video)]['title'].lower()
                    channel = search_results[0].get('uploader', '').lower()
                    duration = search_results[0].get('duration', 0)

                    # Filtering criteria (length <= 5 min, artist or song name in title)
                    if duration <= 300 and (artist_name.lower() in search_results[0]['title'].lower() or song_name.lower() in search_results[0]['title'].lower()):
                        print(f"✅ Downloading '{search_results[0]['title']}' ({duration} seconds)")
                        ydl.download([search_results[0]['webpage_url']])
                        return full_song_path
            except Exception as e:
                print(f"⚠️ Error with query '{query}': {e}")

    raise Exception(f"No suitable song found on YouTube for '{song_name}' by '{artist_name}'.")


def compute_mfcc(audio_path, sr=22050):
    audio, sr = librosa.load(audio_path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    return mfcc

def sliding_window_dtw(full_song_mfcc, snippet_mfcc, hop_length=512, sr=22050):
    snippet_len = snippet_mfcc.shape[1]
    full_len = full_song_mfcc.shape[1]

    step_size = 20  # ~0.2 sec steps
    best_cost = np.inf
    best_idx = 0

    # Normalize snippet once before loop
    snippet_norm = (snippet_mfcc - np.mean(snippet_mfcc)) / (np.std(snippet_mfcc) + 1e-8)

    for idx in range(0, full_len - snippet_len, step_size):
        window = full_song_mfcc[:, idx:idx + snippet_len]
        # Normalize the window similarly
        window_norm = (window - np.mean(window)) / (np.std(window) + 1e-8)
        # Compute DTW with clearly defined distance metric
        D, _ = librosa.sequence.dtw(X=snippet_norm, Y=window_norm, metric='euclidean')
        cost = D[-1, -1]

        if cost < best_cost:
            best_cost = cost
            best_idx = idx

    timestamp_seconds = librosa.frames_to_time(best_idx, sr=sr, hop_length=hop_length)
    return timestamp_seconds, best_cost

def sync_find_preview_start(preview_file, full_song_file):
    if not os.path.exists(full_song_file):
        raise FileNotFoundError(f"File not found: {full_song_file}")

    full_mfcc = compute_mfcc(full_song_file)
    snippet_mfcc = compute_mfcc(preview_file)

    best_timestamp, dtw_cost = sliding_window_dtw(full_mfcc, snippet_mfcc)
    return best_timestamp, dtw_cost

async def find_preview_start(preview_file, full_song_file):
    loop = asyncio.get_event_loop()
    try:
        best_timestamp, dtw_cost = await loop.run_in_executor(
            None, sync_find_preview_start, preview_file, full_song_file
        )

        print(f"✅ DTW match found at {best_timestamp:.2f} seconds (DTW cost: {dtw_cost:.2f})")

        return round(best_timestamp), dtw_cost
    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        return 30, None  # Default guess

async def process_song(ctx, spotify_url):
    match = re.search(SPOTIFY_TRACK_REGEX, spotify_url)
    if not match:
        await ctx.send("❌ Invalid Spotify link.")
        return

    track_id = match.group(1)
    track_info = sp.track(track_id)

    track_name = track_info['name']
    artist_info = sp.artist(track_info['artists'][0]['id'])
    artist_name = track_info['artists'][0]['name']
    genres = artist_info['genres']

    preview_url = get_spotify_preview_url(track_name, artist_name)

    if not preview_url:
        await ctx.send(f"❌ No preview available for **{track_name}** by {artist_name}.")
        return

    song_folder = create_song_folder(track_name, artist_name)

    if song_folder is None:  # Folder already exists, skip processing
        await ctx.send(f"⚠️ Skipping **{track_name}** by {artist_name} (already processed).", delete_after=10)
        return

    preview_file = os.path.join(song_folder, "original_preview.mp3")
    download_audio(preview_url, preview_file)

    full_song_file = download_full_song(track_name, artist_name, song_folder)

    preview_start, dtw_cost = await find_preview_start(preview_file, full_song_file)

    await extract_and_save_preview(preview_start, full_song_file, song_folder)

    song_duration = librosa.get_duration(path=full_song_file)

    metadata_path = os.path.join(song_folder, "metadata.txt")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(f"Song Name: {track_name}\n")
        f.write(f"Artist: {artist_name}\n")
        f.write(f"Detected Timestamp: {preview_start} seconds\n")
        f.write(f"DTW Cost: {dtw_cost:.2f}\n" if dtw_cost else "DTW Cost: Not available\n")
        f.write(f"Full Song Duration: {song_duration:.2f} seconds\n")
        f.write(f"Genres: {', '.join(genres) if genres else 'Unknown'}\n")
