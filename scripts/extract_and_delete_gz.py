#!/usr/bin/env python3

import os
from utils.file_utils import extract_all_gz_in_dir

def extract_recursive(base_dir):
    """
    Recursively searches for and extracts all .csv.gz files in a directory and its subdirectories.

    Args:
        base_dir (str): The base directory to search for .csv.gz files.

    Returns:
        None
    """
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".gz"):
                file_path = os.path.join(root, file)
                print(f"Extracting: {file_path}")
                extract_all_gz_in_dir(root)

if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Extract and delete .csv.gz files from a directory (recursively if needed).")
    parser.add_argument("--dir", type=str, required=True, help="Directory path to extract files from.")

    # Parse arguments
    args = parser.parse_args()
    directory = args.dir

    # Check if the provided directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist.")
        exit(1)

    # Extract all `.csv.gz` files in the specified directory (recursively)
    print(f"Searching and extracting all .csv.gz files in {directory} and its subdirectories...")
    extract_recursive(directory)
    print(f"Extraction complete. All .gz files removed.")


# Usage : Run from woola-bot ( project root) dir
# Extract .gz files from a dir : python3 scripts/extract_and_delete_gz.py --dir data/bybit_data/SOLUSDT/1m-timeframe/
# Extract .gz files from a dir and its subdirs recursively: python3 scripts/extract_and_delete_gz.py --dir data/bybit_data/
