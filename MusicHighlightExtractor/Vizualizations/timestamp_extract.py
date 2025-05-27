import os
import re
import pandas as pd


def extract_fields_from_file(txt_file):
    timestamp = None
    dtw_cost = None

    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            if timestamp is None:
                ts_match = re.search(r'Detected\s+Timestamp:\s*(\d+)\s*seconds?', line, re.IGNORECASE)
                if ts_match:
                    timestamp = int(ts_match.group(1))

            if dtw_cost is None:
                dtw_match = re.search(r'DTW\s+Cost:\s*([\d\.]+)', line, re.IGNORECASE)
                if dtw_match:
                    dtw_cost = float(dtw_match.group(1))

            # Exit early if both are found
            if timestamp is not None and dtw_cost is not None:
                break

    return timestamp, dtw_cost


def collect_oldest_fields(parent_folder, max_folders=100):
    data = []

    subfolders = [
        (f.path, os.path.getctime(f.path))
        for f in os.scandir(parent_folder) if f.is_dir()
    ]

    subfolders.sort(key=lambda x: x[1])
    oldest_subfolders = [folder for folder, _ in subfolders[:max_folders]]

    for subfolder in oldest_subfolders:
        txt_path = os.path.join(subfolder, 'metadata.txt')
        if os.path.isfile(txt_path):
            timestamp, dtw_cost = extract_fields_from_file(txt_path)

            data.append({
                'Subfolder': os.path.basename(subfolder),
                'Detected Timestamp (seconds)': timestamp,
                'DTW Cost': dtw_cost
            })

            if timestamp is None:
                print(f"No timestamp found in '{txt_path}'.")
            if dtw_cost is None:
                print(f"No DTW cost found in '{txt_path}'.")

        else:
            print(f"No metadata.txt found in '{subfolder}'.")

    return data


def save_to_excel(data, output_filename='timestamps_and_dtw_costs.xlsx'):
    if data:
        df = pd.DataFrame(data)
        df.to_excel(output_filename, index=False)
        print(f"Excel file '{output_filename}' saved successfully.")
    else:
        print("No data to save. Excel file was not created.")


# Replace this path with your actual parent folder
parent_folder_path = 'Song_data'

fields_data = collect_oldest_fields(parent_folder_path, max_folders=100)
save_to_excel(fields_data)
