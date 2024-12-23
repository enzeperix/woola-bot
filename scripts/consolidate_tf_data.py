#!/usr/bin/env python3

import os
from utils.file_utils import consolidate_timeframe_data

if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Consolidate CSV files for different timeframes.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing timeframe folders.")
    parser.add_argument("--timeframes", type=str, required=False, default="1m,5m,15m,30m,60m",
                        help="Comma-separated list of timeframe folders (e.g., '1m,5m,15m'). Default: all.")

    # Parse arguments
    args = parser.parse_args()
    base_dir = args.base_dir
    timeframes = args.timeframes.split(",")

    # Consolidate each timeframe
    for timeframe in timeframes:
        timeframe_dir = os.path.join(base_dir, f"{timeframe}-timeframe")
        output_file = os.path.join(base_dir, f"consolidated_{timeframe}-timeframe.csv")
        if os.path.exists(timeframe_dir):
            consolidate_timeframe_data(timeframe_dir, output_file)
        else:
            print(f"Directory not found: {timeframe_dir}")


# Usage:  python3 scripts/consolidate_tf_data.py --base_dir data/bybit_data/SOLUSDT
