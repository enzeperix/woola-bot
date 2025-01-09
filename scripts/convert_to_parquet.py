#!/usr/bin/env python3

import argparse
from utils.file_utils import convert_csv_to_parquet

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Convert a CSV file to Parquet format.")
    parser.add_argument("--file", required=True, help="Path to the CSV file to convert.")
    args = parser.parse_args()

    # Convert CSV to Parquet
    try:
        convert_csv_to_parquet(args.file)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
