# Bot token:
import subprocess
import discord
import re
import spotipy
import librosa
import numpy as np
import requests
import yt_dlp
import os
import asyncio  # Required for async processing
from spotipy.oauth2 import SpotifyClientCredentials
from scipy.signal import correlate
from pydub import AudioSegment
import re

TOKEN = ""
SPOTIFY_CLIENT_ID = ""
SPOTIFY_CLIENT_SECRET = ""
# Initialize Spotify API
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET))

# Initialize Discord bot with intents
intents = discord.Intents.default()
client = discord.Client(intents=intents)

# Regex for Spotify song links
SPOTIFY_TRACK_REGEX = r"https://open\.spotify\.com/track/([a-zA-Z0-9]+)"

# Global queue for song processing
queue = asyncio.Queue()
processing = False  # Flag to indicate if a song is being processed

def sanitize_filename(filename):
    """Removes invalid characters from a filename or folder name."""
    return re.sub(r'[\/:*?"<>|]', '', filename)  # Removes forbidden characters

def create_song_folder(song_name, artist_name):
    """Creates a folder for storing song-related files."""
    folder_name = sanitize_filename(f"{artist_name} - {song_name}")
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

async def extract_and_save_preview(preview_start, full_song_file, song_folder):
    """Extracts the matched preview section from the full song and saves it."""
    try:
        output_file = os.path.join(song_folder, "extracted_preview.mp3")
        song = AudioSegment.from_file(full_song_file, format="mp3")

        # Convert seconds to milliseconds
        start_ms = preview_start * 1000
        end_ms = start_ms + (30 * 1000)  # Extract 30 seconds

        # Extract and save snippet
        snippet = song[start_ms:end_ms]
        snippet.export(output_file, format="mp3")

    except Exception as e:
        print(f"❌ Error extracting preview section: {e}")

def get_spotify_preview_url(song_name):
    """Calls the JavaScript script to get the Spotify preview URL"""
    try:
        result = subprocess.run(["node", "preview_finder.js", song_name], capture_output=True, text=True)
        preview_url = result.stdout.strip()
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

def download_full_song(song_name, song_folder):
    """Download the full song from YouTube and save it in the folder."""
    full_song_path = os.path.join(song_folder, "full_song.mp3")

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'outtmpl': full_song_path[:-4],  # Prevents double .mp3 extension
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"ytsearch1:{song_name}"])  # Searches for the first result and downloads it
    return full_song_path

async def find_preview_start(preview_file, full_song_file):
    """Finds the timestamp where the preview appears in the full song using cross-correlation."""
    try:
        print("🔍 Loading preview audio...")
        y_preview, sr_preview = librosa.load(preview_file, sr=22050)
        y_preview = librosa.util.normalize(y_preview)  # Normalize loudness

        print("🔍 Loading full song audio...")
        y_full, sr_full = librosa.load(full_song_file, sr=22050)
        y_full = librosa.util.normalize(y_full)  # Normalize loudness

        print("🔍 Performing cross-correlation...")
        correlation = correlate(y_full, y_preview, mode='valid')
        best_match = np.argmax(correlation)

        # Convert index to time in seconds
        preview_start_time = best_match / sr_full

        print(f"✅ Best match at {preview_start_time:.2f} seconds")
        return round(preview_start_time)

    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        return 30  # Default guess if processing fails

async def process_queue():
    """Processes the song queue one at a time."""
    global processing
    while True:
        song_data = await queue.get()
        processing = True  # Set processing flag

        message, track_name, artist_name, preview_url, genres = song_data

        await message.channel.send(f"🎵 Now processing **{track_name}** by {artist_name}...")

        # Step 1: Create folder for the song
        song_folder = create_song_folder(track_name, artist_name)

        # Step 2: Download Preview Audio
        preview_file = os.path.join(song_folder, "original_preview.mp3")
        download_audio(preview_url, preview_file)

        # Step 3: Download Full Song
        full_song_file = download_full_song(f"{track_name} {artist_name}", song_folder)

        # Step 4: Find the correct preview timestamp
        preview_start = await find_preview_start(preview_file, full_song_file)

        # Step 5: Save the extracted preview snippet
        await extract_and_save_preview(preview_start, full_song_file, song_folder)

        # Step 6: Save metadata including all genres
        metadata_path = os.path.join(song_folder, "metadata.txt")
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(f"Song Name: {track_name}\n")
            f.write(f"Artist: {artist_name}\n")
            f.write(f"Detected Timestamp: {preview_start} seconds\n")
            f.write(f"Genres: {', '.join(genres) if genres else 'Unknown'}\n")

        # Notify user
        response = f"🎧 Preview found!\n🎵 **{track_name}** by {artist_name}\n🕒 Starts at **{preview_start//60}:{preview_start%60:02}**."
        await message.channel.send(response)

        # Mark task as done
        queue.task_done()
        processing = False  # Reset processing flag

@client.event
async def on_ready():
    print(f'✅ Logged in as {client.user}')
    client.loop.create_task(process_queue())  # Start processing queue

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    match = re.search(SPOTIFY_TRACK_REGEX, message.content)
    if match:
        track_id = match.group(1)
        track_info = sp.track(track_id)

        track_name = track_info['name']
        artist_info = sp.artist(track_info['artists'][0]['id'])  # Fetch artist info separately
        artist_name = track_info['artists'][0]['name']
        genres = artist_info['genres']  # Get all genres from artist

        preview_url = get_spotify_preview_url(track_name)

        if not preview_url:
            await message.channel.send(f"❌ No preview available for **{track_name}** by {artist_name}.")
            return

        # Add the song to the queue
        await queue.put((message, track_name, artist_name, preview_url, genres))

        queue_position = queue.qsize()
        await message.channel.send(f"📌 **{track_name}** by {artist_name} added to the queue. Position: {queue_position+1}")

# Run the bot
client.run(TOKEN)