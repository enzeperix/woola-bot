#!/usr/bin/env python3

import os
import pandas as pd
from utils.define_indicators import calculate_features

def process_dataset(input_path, output_path):
    """
    Loads the dataset, calculates features, and saves the enriched dataset.

    Args:
        input_path (str): Path to the input CSV file.
        output_path (str): Path to save the enriched dataset.

    Returns:
        None
    """
    print(f"Processing {input_path}...")
    # Load input data
    df = pd.read_csv(input_path, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Generate features
    df_with_features = calculate_features(df)

    # Save to output
    df_with_features.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")

if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate features for OHLCV data.")
    parser.add_argument("--input", type=str, required=True, nargs='+',
                        help="One or more input consolidated CSV files.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save enriched datasets.")

    # Parse arguments
    args = parser.parse_args()
    input_files = args.input
    output_dir = args.output_dir

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each input file
    for input_file in input_files:
        base_name = os.path.basename(input_file).replace("consolidated_", "").replace(".csv", "_with_indicators.csv")
        output_file = os.path.join(output_dir, base_name)
        process_dataset(input_file, output_file)

# To generate features for a consolidated CSV file:
# python3 scripts/compute_indicators.py --input data/bybit_data/SOLUSDT/consolidated_1m-timeframe.csv --output_dir data/bybit_data/SOLUSDT/enriched/



