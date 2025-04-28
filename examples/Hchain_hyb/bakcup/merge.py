import os
import json
from tqdm import tqdm
import argparse

def merge_json_files(input_folder, output_file):
    merged_data = {}

    # List all .json files
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

    # Use tqdm to show progress
    for filename in tqdm(json_files, desc="Merging JSON files"):
        filepath = os.path.join(input_folder, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                key = os.path.splitext(filename)[0]  # filename without .json
                merged_data[key] = data
            except json.JSONDecodeError:
                print(f"Warning: {filename} is not a valid JSON file. Skipped.")

    # Write the merged dictionary to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2)
        
def batch_merge(x):
    merge_folder = f"{x}_merge"
    os.makedirs(merge_folder, exist_ok=True)

    merge_json_files(f"{x}/obs", f"{merge_folder}/obs.json")
    merge_json_files(f"{x}/basic", f"{merge_folder}/basic.json")

def main():
    parser = argparse.ArgumentParser(description="Merge JSON files from a given folder")
    parser.add_argument('--folder', type=str, required=True, help='Input folder name (x)')
    args = parser.parse_args()

    batch_merge(args.folder)

if __name__ == "__main__":
    main()