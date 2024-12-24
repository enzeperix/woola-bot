#!/usr/bin/env python3

import os
from utils.file_utils import consolidate_timeframe_data

def process_directory(base_dir, timeframes):
    """
    Consolidates CSV files for the specified timeframes in a given base directory.

    Args:
        base_dir (str): The base directory to process.
        timeframes (list): List of timeframe folders to consolidate.

    Returns:
        None
    """
    for timeframe in timeframes:
        timeframe_dir = os.path.join(base_dir, f"{timeframe}-timeframe")
        output_file = os.path.join(base_dir, f"consolidated_{timeframe}-timeframe.csv")
        if os.path.exists(timeframe_dir):
            print(f"Processing {timeframe_dir}...")
            consolidate_timeframe_data(timeframe_dir, output_file)
        else:
            print(f"Timeframe directory not found: {timeframe_dir}")


def process_recursive(base_dir, timeframes):
    """
    Recursively consolidates data for all pair directories under a given base directory.

    Args:
        base_dir (str): The top-level directory to process recursively.
        timeframes (list): List of timeframe folders to consolidate.

    Returns:
        None
    """
    for root, dirs, _ in os.walk(base_dir):
        for d in dirs:
            pair_dir = os.path.join(root, d)
            print(f"Found pair directory: {pair_dir}")
            process_directory(pair_dir, timeframes)


if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Consolidate CSV files for different timeframes.")
    parser.add_argument("--dir", type=str, required=True, nargs='+',
                        help="One or more directories to process. For recursive processing, provide a base directory.")
    parser.add_argument("--timeframes", type=str, required=False, default="1m,5m,15m,30m,60m",
                        help="Comma-separated list of timeframe folders (e.g., '1m,5m,15m'). Default: all.")

    # Parse arguments
    args = parser.parse_args()
    dirs = args.dir
    timeframes = args.timeframes.split(",")

    # Process each directory or recursively process the base directory
    for directory in dirs:
        if os.path.exists(directory):
            print(f"Processing directory: {directory}")
            if any(os.path.isdir(os.path.join(directory, sub)) for sub in os.listdir(directory)):
                # If the directory contains subdirectories, process recursively
                process_recursive(directory, timeframes)
            else:
                # Otherwise, treat it as a single pair directory
                process_directory(directory, timeframes)
        else:
            print(f"Directory does not exist: {directory}")


# Usage: Run from root dir of the repo
# Process a single dir: python3 scripts/consolidate_tf_data.py --dir data/bybit_data/SOLUSDT
# process 2 or more dirs: python3 scripts/consolidate_tf_data.py --dir data/bybit_data/SOLUSDT data/bybit_data/NEARUSDT
# Process a dir and its subdirs recursively: python3 scripts/consolidate_tf_data.py --dir data/bybit_data