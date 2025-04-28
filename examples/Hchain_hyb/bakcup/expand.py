import os
import json
import argparse
from tqdm import tqdm

def expand_json_file(input_file, output_folder):
    # Load the big merged JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        merged_data = json.load(f)

    os.makedirs(output_folder, exist_ok=True)

    # For each key-value pair, save as its own .json file
    for filename, content in tqdm(merged_data.items(), desc=f"Expanding {input_file}"):
        output_path = os.path.join(output_folder, f"{filename}.json")
        with open(output_path, 'w', encoding='utf-8') as f_out:
            json.dump(content, f_out, indent=2)

def batch_expand(x):
    expand_folder = f"{x}_expand"
    os.makedirs(expand_folder, exist_ok=True)

    expand_json_file(os.path.join(x, "obs.json"), os.path.join(expand_folder, "obs"))
    expand_json_file(os.path.join(x, "basic.json"), os.path.join(expand_folder, "basic"))

def main():
    parser = argparse.ArgumentParser(description="Expand merged JSON files into individual JSON files")
    parser.add_argument('--folder', type=str, required=True, help='Input merged folder name (e.g., x_merge)')
    args = parser.parse_args()

    batch_expand(args.folder)

if __name__ == "__main__":
    main()
