#!/usr/bin/env python3

import os
import requests
import argparse

def download_bybit_csv(pair, year, timeframes, root_dir="../data"):
    """
    Downloads ByBit historical candlestick data from their public repository.

    Args:
        pair (str): Trading pair, e.g., 'SOLUSDT'.
        year (int): Year to download data for, e.g., 2021.
        timeframes (list): List of timeframes to download, e.g., [1, 5, 15, 30, 60].
        root_dir (str): Root directory where the data will be stored. Defaults to '../data'.

    Returns:
        None
    """
    # Root folder for all data
    output_root = os.path.join(root_dir, "bybit_data", pair)
    os.makedirs(output_root, exist_ok=True)

    base_url = f"https://public.bybit.com/kline_for_metatrader4/{pair}/{year}/"
    
    for timeframe in timeframes:
        # Create subfolder for each timeframe
        timeframe_folder = os.path.join(output_root, f"{timeframe}m-timeframe")
        os.makedirs(timeframe_folder, exist_ok=True)

        for month in range(1, 13):
            month_str = f"{month:02d}"  # Format month as two digits (01, 02, etc.)
            end_day = "30" if month in [4, 6, 9, 11] else "31" if month != 2 else "28"
            file_name = f"{pair}_{timeframe}_{year}-{month_str}-01_{year}-{month_str}-{end_day}.csv.gz"
            file_url = f"{base_url}{file_name}"
            output_file = os.path.join(timeframe_folder, file_name)
            
            try:
                print(f"Downloading {file_name}...")
                response = requests.get(file_url, stream=True)
                if response.status_code == 200:
                    with open(output_file, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"Saved to {output_file}")
                else:
                    print(f"File not found: {file_name}")
            except Exception as e:
                print(f"Error downloading {file_name}: {e}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download ByBit historical candlestick data.")
    parser.add_argument("--pair", type=str, required=True, help="Trading pair, e.g., 'SOLUSDT'.")
    parser.add_argument("--year", type=int, required=True, help="Year to download data for, e.g., 2021.")
    parser.add_argument("--timeframes", type=str, required=True, help="Comma-separated list of timeframes, e.g., '1,5,15,30,60'.")
    parser.add_argument("--root_dir", type=str, default="../data", help="Root directory to save the files. Default: '../data'.")

    # Parse arguments
    args = parser.parse_args()
    pair = args.pair
    year = args.year
    timeframes = [int(t.strip()) for t in args.timeframes.split(",")]
    root_dir = args.root_dir

    # Run the downloader
    download_bybit_csv(pair, year, timeframes, root_dir)
