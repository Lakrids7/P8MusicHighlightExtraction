import csv
import os
import argparse
import yt_dlp  # Preferred over youtube_dl


def download_youtube_audio_as_mp3(youtube_id, salami_id, output_dir, audio_quality="192",
                                  archive_file="download_archive.txt"):
    """
    Downloads audio from a given YouTube ID and saves it as an MP3.

    Args:
        youtube_id (str): The YouTube video ID.
        salami_id (str): The SALAMI ID, used for naming the output file.
        output_dir (str): The directory to save the downloaded audio.
        audio_quality (str): The desired audio quality for the MP3 (e.g., '192', '320').
        archive_file (str): Path to a file to record downloaded video IDs to avoid re-downloading.
    Returns:
        bool: True if download was successful or skipped (already in archive), False otherwise.
    """
    if not youtube_id or youtube_id == '0' or len(youtube_id) < 5:  # Basic validation for YouTube ID
        print(f"Skipping Salami ID {salami_id}: Invalid or missing YouTube ID '{youtube_id}'")
        return False

    video_url = f"https://www.youtube.com/watch?v={youtube_id}"

    # Construct filename template. yt-dlp will set extension to .mp3 after postprocessing.
    output_template = os.path.join(output_dir, f"salami_{salami_id}_{youtube_id}.%(ext)s")

    ydl_opts = {
        'format': 'bestaudio/best',  # Download best audio-only format
        'outtmpl': output_template,
        'noplaylist': True,  # Only download single video, not playlist
        'quiet': True,  # Suppress yt-dlp's own console output
        'ignoreerrors': False,  # Raise exception on error for this video
        'download_archive': archive_file,  # File to keep track of downloaded videos
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',  # Ensure output is MP3
            'preferredquality': audio_quality,
        }]
    }

    print(f"Attempting Salami ID: {salami_id}, YouTube ID: {youtube_id} (URL: {video_url}) as MP3")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.download([video_url])
            if result == 0 or result is None:  # None can happen if skipped by archive
                print(f"  -> Success/Skipped (already downloaded): Salami ID: {salami_id}")
                return True
            else:
                print(f"  -> yt-dlp indicated an issue (result code: {result}) for Salami ID: {salami_id}")
                return False

    except yt_dlp.utils.DownloadError as e:
        if "has already been recorded in the archive" in str(e) or "already been downloaded" in str(e):
            print(f"  -> Skipped (already in archive): Salami ID: {salami_id}")
            return True
        print(f"  -> ERROR downloading Salami ID {salami_id} (YT: {youtube_id}): {str(e).splitlines()[0]}")
        return False
    except Exception as e:
        print(f"  -> UNEXPECTED ERROR for Salami ID {salami_id} (YT: {youtube_id}): {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download YouTube audio tracks as MP3s from a SALAMI CSV file.")
    parser.add_argument("csv_file", help="Path to the input CSV file (e.g., salami_youtube_pairings.csv)")
    parser.add_argument("-o", "--output-dir", default="salami_mp3_downloads",
                        help="Directory to save downloaded MP3 files (default: salami_mp3_downloads)")
    parser.add_argument("-q", "--quality", default="192",
                        help="Preferred MP3 audio quality (e.g., 128, 192, 320; default: 192).")
    parser.add_argument("--archive-file", default="download_mp3_archive.txt",
                        help="File to keep track of downloaded videos to avoid re-downloading (default: download_mp3_archive.txt)")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of songs to process (for testing).")

    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' not found.")
        return

    if not os.path.exists(args.output_dir):
        print(f"Creating output directory: {args.output_dir}")
        os.makedirs(args.output_dir)

    successful_downloads = 0
    failed_downloads = 0
    skipped_invalid = 0
    total_processed = 0

    try:
        with open(args.csv_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            if 'salami_id' not in reader.fieldnames or 'youtube_id' not in reader.fieldnames:
                print("Error: CSV file must contain 'salami_id' and 'youtube_id' columns.")
                return

            for i, row in enumerate(reader):
                if args.limit and total_processed >= args.limit:
                    print(f"Reached download limit of {args.limit}.")
                    break

                total_processed += 1
                salami_id = row.get('salami_id')
                youtube_id = row.get('youtube_id')

                if not salami_id or not youtube_id:
                    print(
                        f"Skipping row {i + 2} (CSV line number) due to missing salami_id or youtube_id: {row}")  # i+2 because header is 1, first data row is 2
                    skipped_invalid += 1
                    continue

                if download_youtube_audio_as_mp3(youtube_id, salami_id, args.output_dir, args.quality,
                                                 args.archive_file):
                    successful_downloads += 1
                else:
                    # Only count as failed if it was a valid attempt and not skipped for invalid ID
                    if youtube_id and youtube_id != '0' and len(youtube_id) >= 5:
                        failed_downloads += 1
                    else:
                        skipped_invalid += 1

    except FileNotFoundError:
        print(f"Error: CSV file '{args.csv_file}' not found.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during CSV processing: {e}")
        return
    finally:
        print("\n--- MP3 Download Summary ---")
        print(f"Total entries processed from CSV: {total_processed}")
        print(f"Successfully downloaded/skipped (already done): {successful_downloads}")
        print(f"Failed to download (actual errors): {failed_downloads}")
        print(f"Skipped due to invalid/missing IDs in CSV: {skipped_invalid}")
        print(f"MP3 files saved in: {args.output_dir}")
        print(f"Download archive (to prevent re-downloads) at: {args.archive_file}")


if __name__ == "__main__":
    main()