#!/usr/bin/env python3

import argparse
import logging
import pandas as pd
from utils.file_utils import save_factors_to_csv
from utils.factors_utils import (
    calculate_mid_prices,
    generate_expanded_differences,
    generate_ratios,
    generate_pairwise_operations,
    generate_rolling_statistics
)

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def generate_factors_universe(input_csv):
    """
    Generates the factors universe from the input DataFrame and saves it to a CSV.
    Args:
        input_csv (str): Path to the input CSV file.
    """
    try:
        # Load the dataset
        logging.info("Step 0: Loading the input dataset.")
        df = pd.read_csv(input_csv, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        logging.info(f"Step 0: Dataset loaded with {len(df)} rows.")

        # Step 1: Add mid-prices
        logging.info("Step 1: Adding mid-prices to the dataset.")
        df = calculate_mid_prices(df)

        # Step 2: Generate Simple Differences
#        logging.info("Step 2: Generating expanded differences.")
#        df = generate_expanded_differences(df)

        # Step 3: Generate Ratios
#        logging.info("Step 3: Generating ratios.")
#        df = generate_ratios(df)
#
#        # Step 4: Generate Pairwise Operations
        logging.info("Step 4: Generating pairwise operations.")
        df = generate_pairwise_operations(df)
#
#        # Step 5: Generate Rolling Statistics
#        logging.info("Step 5: Generating rolling statistics.")
#        df = generate_rolling_statistics(df)
#
        # Save the generated factors universe
        logging.info("Step 6: Saving the factors universe to a CSV file.")
        save_factors_to_csv(df, input_csv)

        logging.info("Factors universe generation completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred while generating the factors universe: {e}")
        raise

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate factors universe from OHLCV data.")
    parser.add_argument("input_csv", help="Path to the input CSV file containing OHLCV data")
    args = parser.parse_args()

    # Run the factor generation pipeline
    generate_factors_universe(args.input_csv)
