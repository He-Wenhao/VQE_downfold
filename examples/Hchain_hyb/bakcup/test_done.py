import os
import argparse
from tqdm import tqdm

def check_folders(root_folder, log_filename, keyword):
    unsatisfied_folders = []

    all_folders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]

    for folder_name in tqdm(all_folders, desc="Checking folders"):
        folder_path = os.path.join(root_folder, folder_name)
        log_file = os.path.join(folder_path, log_filename)

        if not os.path.exists(log_file):
            unsatisfied_folders.append(folder_name)
            continue

        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                found = any(keyword in line for line in lines)
                if not found:
                    unsatisfied_folders.append(folder_name)
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
            unsatisfied_folders.append(folder_name)

    return unsatisfied_folders

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str, default='./H6', help='Root folder containing subfolders')
    parser.add_argument('--log_filename', type=str, default='l_opt_log.txt', help='Log filename to check')
    parser.add_argument('--keyword', type=str, default='total running time', help='Keyword to search for in the log file')
    args = parser.parse_args()

    unsatisfied = check_folders(args.root_folder, args.log_filename, args.keyword)

    print("\nUnsatisfied folders:")
    for folder in unsatisfied:
        print(folder)

