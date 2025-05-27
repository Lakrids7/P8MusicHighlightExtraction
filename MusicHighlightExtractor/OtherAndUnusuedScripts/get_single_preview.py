import subprocess
import re
import os
import requests
import argparse
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import librosa # <--- Import librosa

# --- Spotify API Setup ---
SPOTIFY_CLIENT_ID = "f417419a96c14de0a9a7a1524b8b8184" # Replace if needed
SPOTIFY_CLIENT_SECRET = "79d04c715de94c43989eb4215b657be9" # Replace if needed

try:
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID,
                                                               client_secret=SPOTIFY_CLIENT_SECRET))
except Exception as e:
    print(f"❌ Error initializing Spotify client: {e}")
    sp = None

# --- Regex for Spotify Track URL ---
SPOTIFY_TRACK_REGEX = r"https?://open\.spotify\.com/(intl-\w+/)?track/([a-zA-Z0-9]+)"

# --- Reusable Helper Functions ---

def get_track_id_from_url(url):
    """Extracts the Spotify Track ID from a URL."""
    match = re.search(SPOTIFY_TRACK_REGEX, url)
    return match.group(2) if match else None

def get_track_info(track_id):
    """Fetches track name and primary artist name using Spotipy."""
    if not sp:
        print("❌ Cannot fetch track info: Spotify client not initialized.")
        return None, None
    try:
        print(f"ℹ️ Fetching track info for ID: {track_id}...")
        track_info = sp.track(track_id)
        if not track_info:
            print(f"❌ No track info found for ID: {track_id}")
            return None, None

        track_name = track_info.get('name')
        artists = track_info.get('artists')
        if not track_name or not artists:
            print(f"❌ Incomplete track info received for ID: {track_id}")
            return None, None
        artist_name = artists[0].get('name')
        if not artist_name:
             print(f"❌ Could not get artist name for ID: {track_id}")
             return None, None

        print(f"✅ Found Track: '{track_name}' by '{artist_name}'")
        return track_name, artist_name

    except spotipy.exceptions.SpotifyException as e:
        print(f"❌ Spotify API error fetching track info: {e}")
        return None, None
    except Exception as e:
        print(f"❌ Unexpected error fetching track info: {e}")
        return None, None

def sanitize_filename(filename):
    """Removes invalid characters from a filename."""
    if not filename: return "unknown"
    filename = str(filename)
    filename = re.sub(r'[\\/:*?"<>|#\n\r\t]', '', filename)
    filename = re.sub(r'[\s_]+', '_', filename)
    filename = filename.strip('_. ')
    if not filename:
        filename = "untitled"
    return filename

def get_spotify_preview_url(song_name, artist_name):
    """Uses the external 'preview_finder.js' Node.js script."""
    if not song_name or not artist_name:
        print("❌ Cannot find preview URL without song name and artist name.")
        return None
    query = f"{song_name} {artist_name}"
    print(f"🔍 Searching for preview: '{query}' using preview_finder.js...")
    try:
        script_path = os.path.join(os.path.dirname(__file__), "preview_finder.js")
        if not os.path.exists(script_path):
             script_path = "preview_finder.js"
        result = subprocess.run(
            ["node", script_path, query],
            capture_output=True, text=True, check=False, encoding='utf-8'
        )
        if result.returncode != 0:
            print(f"❌ Error running Node.js script (RC: {result.returncode}):\n   Stderr: {result.stderr.strip()}")
            return None
        preview_url = result.stdout.strip()
        if not preview_url or "No preview available" in preview_url or not preview_url.startswith("http"):
            print(f"ℹ️ No valid preview URL found by Node.js script.")
            return None
        print(f"🔗 Found Spotify preview URL: {preview_url}")
        return preview_url
    except FileNotFoundError:
        print(f"❌ Error: 'node' command not found. Is Node.js installed and in PATH?")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred running Node.js script: {e}")
        return None

def download_audio(url, output_filename):
    """Download an audio file from a given URL."""
    print(f"⬇️ Downloading preview to '{os.path.basename(output_filename)}'...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(output_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Preview downloaded successfully!")
        return output_filename # Return path on success
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading audio from {url}: {e}")
        if os.path.exists(output_filename):
            try: os.remove(output_filename)
            except OSError: pass
        return None # Return None on failure
    except Exception as e:
        print(f"❌ An unexpected error occurred during download: {e}")
        if os.path.exists(output_filename):
             try: os.remove(output_filename)
             except OSError: pass
        return None # Return None on failure

# --- NEW Function ---
def get_audio_duration(filepath):
    """Measures the duration of an audio file using librosa."""
    if not os.path.exists(filepath):
        print(f"⚠️ Cannot measure duration: File not found at '{filepath}'")
        return None
    try:
        duration = librosa.get_duration(path=filepath)
        return duration
    except Exception as e:
        print(f"⚠️ Error measuring duration for '{filepath}': {e}")
        return None

# --- Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a Spotify track preview and print its duration.")
    parser.add_argument("--url", type=str, required=True, help="The full Spotify track URL.")
    parser.add_argument("--output_dir", type=str, default="downloaded_previews", help="Directory to save the downloaded preview file.")

    args = parser.parse_args()

    # --- Workflow ---
    print("--- Spotify Preview Downloader ---")
    # 1. Validate Spotify Client
    if not sp:
        exit(1)

    # 2. Extract Track ID
    print(f"\n1. Parsing URL: {args.url}")
    track_id = get_track_id_from_url(args.url)
    if not track_id:
        print(f"❌ Invalid Spotify track URL format provided.")
        exit(1)
    print(f"   Extracted Track ID: {track_id}")

    # 3. Get Track Info
    print("\n2. Fetching Track Metadata...")
    song_name, artist_name = get_track_info(track_id)
    if not song_name or not artist_name:
        exit(1)

    # 4. Find Preview URL
    print("\n3. Finding Preview MP3 URL...")
    preview_url = get_spotify_preview_url(song_name, artist_name)
    if not preview_url:
        exit(1)

    # 5. Prepare Download Path
    print("\n4. Preparing Download...")
    sanitized_song = sanitize_filename(song_name)
    sanitized_artist = sanitize_filename(artist_name)
    output_filename_base = f"{sanitized_artist}_{sanitized_song}_preview.mp3"
    os.makedirs(args.output_dir, exist_ok=True)
    output_filepath = os.path.join(args.output_dir, output_filename_base)
    print(f"   Output path: {output_filepath}")

    # 6. Download Audio
    print("\n5. Downloading Audio...")
    downloaded_path = download_audio(preview_url, output_filepath) # Get path if successful

    if not downloaded_path:
        print(f"\n❌ Failed to download preview for '{song_name}'.")
        exit(1)

    # 7. Measure and Print Duration <--- NEW STEP ---
    print("\n6. Measuring Duration...")
    duration = get_audio_duration(downloaded_path)

    if duration is not None:
        print(f"   ⏱️ Duration of downloaded preview: {duration:.2f} seconds")
    else:
        print(f"   ⚠️ Could not determine duration of the downloaded file.") # Error already printed by get_audio_duration

    print(f"\n🎉 Successfully processed preview for '{song_name}'")
    print(f"   File saved to: '{downloaded_path}'")