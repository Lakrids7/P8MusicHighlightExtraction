import os
import shutil

SOURCE_DIR = "Song_data(CopyrightFree)"
DEST_DIR = "Filtered_Songs"
DTW_THRESHOLD = 1000.0

def extract_dtw_cost(metadata_path):
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("DTW Cost:"):
                    return float(line.split(":")[1].strip())
    except UnicodeDecodeError:
        print(f"⚠️ Unicode decode error in file: {metadata_path}. Trying fallback encoding.")
        try:
            with open(metadata_path, "r", encoding="latin1") as f:
                for line in f:
                    if line.startswith("DTW Cost:"):
                        return float(line.split(":")[1].strip())
        except Exception as e:
            print(f"❌ Failed to read file with fallback encoding: {metadata_path}\n{e}")
    except Exception as e:
        print(f"❌ Error reading DTW cost from {metadata_path}: {e}")
    return None

def main():
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    for subfolder in os.listdir(SOURCE_DIR):
        subfolder_path = os.path.join(SOURCE_DIR, subfolder)
        metadata_path = os.path.join(subfolder_path, "metadata.txt")
        destination = os.path.join(DEST_DIR, subfolder)

        if os.path.isdir(subfolder_path) and os.path.exists(metadata_path):
            if os.path.exists(destination):
                print(f"🟡 Already exists, skipping: {subfolder}")
                continue

            dtw_cost = extract_dtw_cost(metadata_path)
            if dtw_cost is not None and dtw_cost < DTW_THRESHOLD:
                shutil.copytree(subfolder_path, destination)
                print(f"✅ Copied: {subfolder} (DTW Cost: {dtw_cost})")
            else:
                print(f"⏭️ Skipped: {subfolder} (DTW Cost: {dtw_cost})")
        else:
            print(f"❓ No metadata.txt found in: {subfolder_path}")


if __name__ == "__main__":
    main()
