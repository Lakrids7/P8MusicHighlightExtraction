import os
import re
import argparse
import librosa
import math
import sys
from collections import OrderedDict # To preserve metadata order better

def parse_metadata_ordered(filepath):
    """Parses existing metadata.txt into an ordered dictionary."""
    metadata = OrderedDict() # Use OrderedDict to keep original order
    if not os.path.exists(filepath):
        print(f"  ❌ Metadata file not found: {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                match = re.match(r'([^:]+):\s*(.*)', line)
                if match:
                    key = match.group(1).strip()
                    value = match.group(2).strip()
                    metadata[key] = value
    except Exception as e:
        print(f"  ❌ Error reading metadata file {filepath}: {e}")
        return None
    # Ensure essential key exists, even if empty, for later logic
    if "Detected Timestamp" not in metadata:
         print(f"  ⚠️ 'Detected Timestamp' key missing in {filepath}. Cannot calculate end time.")
         return None # Cannot proceed without start time
    return metadata

def get_audio_duration(filepath):
    """Safely gets audio duration using librosa."""
    if not os.path.exists(filepath):
        print(f"  ⚠️ Audio file not found for duration calculation: {os.path.basename(filepath)}")
        return None
    try:
        duration = librosa.get_duration(filename=filepath)
        return duration
    except Exception as e:
        print(f"  ❌ Error getting duration for {os.path.basename(filepath)}: {e}")
        return None

def format_duration(duration_sec):
    """Formats duration in seconds to 'XXX.XX seconds'."""
    if duration_sec is None:
        return "Unknown"
    try:
        return f"{duration_sec:.2f} seconds"
    except Exception:
        return "Unknown"

def process_folder_add_duration(folder_path):
    """
    Processes a single song folder to add preview duration and end timestamp
    to the metadata.txt, preserving other existing fields.
    """
    print(f"\nProcessing folder: {os.path.basename(folder_path)}")
    metadata_file = os.path.join(folder_path, "metadata.txt")
    preview_file = os.path.join(folder_path, "original_preview.mp3") # Needed for duration

    # --- Check required files ---
    if not os.path.exists(metadata_file):
        print("  ⚠️ metadata.txt not found. Skipping.")
        return False
    if not os.path.exists(preview_file):
        print("  ⚠️ original_preview.mp3 not found. Cannot calculate preview duration/end time. Skipping.")
        return False

    # --- Parse Existing Data (Ordered) ---
    print("  🔍 Reading existing metadata...")
    old_metadata = parse_metadata_ordered(metadata_file)
    if old_metadata is None:
        # Error message already printed in parse_metadata_ordered
        return False

    # --- Check if fields already exist ---
    if "Preview Duration" in old_metadata and "Preview End Timestamp" in old_metadata:
        print("  ℹ️ Preview Duration and End Timestamp already exist. Skipping.")
        return True # Consider this a success/no-op

    # --- Calculate Preview Duration ---
    print("  ⏱️ Calculating preview duration...")
    preview_duration = get_audio_duration(preview_file)

    if preview_duration is None:
        print("  ❌ Failed to get preview duration. Cannot calculate end time or duration. Skipping update.")
        return False

    print(f"    - Calculated Preview Duration: {preview_duration:.2f}s")

    # --- Calculate End Timestamp ---
    detected_ts_str = old_metadata.get("Detected Timestamp", "0 seconds") # Already checked for key existence
    try:
        ts_match = re.search(r'(\d+)', detected_ts_str) # Find first number
        detected_start_sec = int(ts_match.group(1)) if ts_match else 0
    except (ValueError, TypeError, AttributeError):
        print(f"  ⚠️ Could not parse 'Detected Timestamp' value: '{detected_ts_str}'. Using 0s for end time calculation.")
        detected_start_sec = 0

    preview_end_time = detected_start_sec + preview_duration
    preview_end_time_rounded = math.ceil(preview_end_time) # Round up

    print(f"    - Original Start Timestamp: {detected_start_sec}s")
    print(f"    - Calculated Preview End Time: {preview_end_time_rounded}s")

    # --- Format New Values ---
    formatted_preview_duration = format_duration(preview_duration)
    formatted_end_timestamp = f"{preview_end_time_rounded} seconds"

    # --- Construct New Metadata Content (Inserting new fields) ---
    print("  ✍️ Preparing updated metadata...")
    new_metadata = OrderedDict()
    inserted = False
    for key, value in old_metadata.items():
        new_metadata[key] = value
        # Insert the new fields immediately after "Detected Timestamp"
        if key == "Detected Timestamp" and not inserted:
            # Only add if they weren't already present (redundant check, but safe)
            if "Preview End Timestamp" not in old_metadata:
                 new_metadata["Preview End Timestamp"] = formatted_end_timestamp
                 print(f"    + Adding Preview End Timestamp: {formatted_end_timestamp}")
            if "Preview Duration" not in old_metadata:
                 new_metadata["Preview Duration"] = formatted_preview_duration
                 print(f"    + Adding Preview Duration: {formatted_preview_duration}")
            inserted = True

    # Fallback: If "Detected Timestamp" wasn't found (shouldn't happen due to earlier check),
    # or if loop finished without inserting, add them now (less ideal order).
    if not inserted:
         if "Preview End Timestamp" not in new_metadata:
            new_metadata["Preview End Timestamp"] = formatted_end_timestamp
            print(f"    + Adding Preview End Timestamp (fallback order): {formatted_end_timestamp}")
         if "Preview Duration" not in new_metadata:
            new_metadata["Preview Duration"] = formatted_preview_duration
            print(f"    + Adding Preview Duration (fallback order): {formatted_preview_duration}")

    # --- Write Updated Metadata ---
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            for key, value in new_metadata.items():
                f.write(f"{key}: {value}\n")
        print(f"  ✅ Successfully updated {metadata_file} with duration info.")
        return True
    except Exception as e:
        print(f"  ❌ Error writing updated metadata to {metadata_file}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Add Preview Duration and End Timestamp to existing metadata.txt files.")
    parser.add_argument("data_folder", help="Path to the main directory containing song subfolders (e.g., Song_data_New)")
    args = parser.parse_args()

    parent_folder = args.data_folder
    if not os.path.isdir(parent_folder):
        print(f"Error: Folder not found: {parent_folder}")
        sys.exit(1)

    print(f"Starting metadata update process (adding duration/end time only) in: {parent_folder}")
    updated_count = 0
    skipped_count = 0
    already_present_count = 0

    for item_name in sorted(os.listdir(parent_folder)): # Sort for predictable order
        item_path = os.path.join(parent_folder, item_name)
        if os.path.isdir(item_path):
            result = process_folder_add_duration(item_path)
            if result is True:
                 # Check if it was skipped because data was already there
                 metadata_path_check = os.path.join(item_path, "metadata.txt")
                 if os.path.exists(metadata_path_check):
                      with open(metadata_path_check, 'r', encoding='utf-8') as f_check:
                           content = f_check.read()
                           if "Preview Duration:" in content and "Preview End Timestamp:" in content:
                                already_present_count +=1
                           else: # This case should ideally not happen if result was True
                                updated_count += 1
                 else: # Should not happen if result is True
                    updated_count += 1

            elif result is False:
                 skipped_count += 1
            # If result is None (not used here, but good practice), treat as skip/error

    print("\n--------------------")
    print("Metadata update (add duration/end time) complete.")
    print(f"Folders successfully updated: {updated_count}")
    print(f"Folders already containing duration/end time: {already_present_count}")
    print(f"Folders skipped due to errors or missing files: {skipped_count}")
    print("--------------------")

if __name__ == "__main__":
    # Ensure necessary libraries are installed: pip install librosa
    main()