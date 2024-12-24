#!/usr/bin/env python3

import os
import requests
import argparse
from utils.file_utils import extract_all_gz_in_dir
from bs4 import BeautifulSoup

def list_available_years(pair):
    """
    Lists available years for a given trading pair from the ByBit public directory.

    Args:
        pair (str): Trading pair, e.g., 'SOLUSDT'.

    Returns:
        list: A list of available years as integers.
    """
    base_url = f"https://public.bybit.com/kline_for_metatrader4/{pair}/"
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            # Extract year folders from the HTML response
            soup = BeautifulSoup(response.text, "html.parser")
            years = [int(link.text.strip("/")) for link in soup.find_all("a") if link.text.strip("/").isdigit()]
            return years
        else:
            print(f"Failed to list available years for {pair}. HTTP Status: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error while fetching available years: {e}")
        return []


def download_bybit_csv(pair, years, timeframes, root_dir="../data"):
    """
    Downloads ByBit historical candlestick data for multiple years and pairs from their public repository.

    Args:
        pair (str): Trading pair, e.g., 'SOLUSDT'.
        years (list): List of years to download data for, e.g., [2021, 2022].
        timeframes (list): List of timeframes to download, e.g., [1, 5, 15, 30, 60].
        root_dir (str): Root directory where the data will be stored. Defaults to '../data'.

    Returns:
        None
    """
    # Root folder for all data
    output_root = os.path.join(root_dir, "bybit_data", pair)
    os.makedirs(output_root, exist_ok=True)

    for year in years:
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
                extracted_file = output_file[:-3]  # Remove `.gz` extension

                # Skip downloading if the file already exists (either .csv.gz or extracted .csv)
                if os.path.exists(output_file) or os.path.exists(extracted_file):
                    print(f"File already exists, skipping: {extracted_file}")
                    continue

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
    parser.add_argument("--pairs", type=str, required=True, help="Comma-separated list of trading pairs, e.g., 'SOLUSDT,BTCUSDT'.")
    parser.add_argument("--year", type=str, required=True,
                        help="Year(s) to download data for, e.g., '2021', '2021,2022' or 'all'.")
    parser.add_argument("--timeframes", type=str, required=True,
                        help="Comma-separated list of timeframes, e.g., '1,5,15,30,60'.")
    parser.add_argument("--root_dir", type=str, default="../data", help="Root directory to save the files. Default: '../data'.")

    # Parse arguments
    args = parser.parse_args()
    pairs = args.pairs.split(",")
    year_arg = args.year
    timeframes = [int(t.strip()) for t in args.timeframes.split(",")]
    root_dir = args.root_dir

    # Determine the years to download
    for pair in pairs:
        if year_arg.lower() == "all":
            print(f"Fetching available years for pair {pair}...")
            years = list_available_years(pair)
            if not years:
                print(f"No available years found for {pair}. Skipping.")
                continue
        else:
            years = [int(y.strip()) for y in year_arg.split(",")]

        # Run the downloader for the pair
        download_bybit_csv(pair, years, timeframes, root_dir)
# Usage :  Run from data_retrieval directory
# Example: python3 bybit.py --pairs AVAXUSDT,ETHUSDT,NEARUSDT --year all --timeframes 1,5,15,30,60
 