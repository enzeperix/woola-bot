#!/usr/bin/env python3


import os
from utils.file_utils import extract_all_gz_in_dir

if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Extract and delete .csv.gz files from a directory.")
    parser.add_argument("--dir", type=str, required=True, help="Directory path to extract files from.")

    # Parse arguments
    args = parser.parse_args()
    directory = args.dir

    # Extract all `.csv.gz` files in the specified directory
    print(f"Extracting all .csv.gz files in {directory}...")
    extract_all_gz_in_dir(directory)
    print(f"Extraction complete. All .gz files removed.")
