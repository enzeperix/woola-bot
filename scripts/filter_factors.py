import argparse
import logging
import pandas as pd
from utils.filtering_utils import filter_features_by_correlation_dask

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

def save_to_file(df, output_file, output_type):
    """
    Saves a DataFrame to CSV or Parquet based on the output type.
    Args:
        df (pd.DataFrame): DataFrame to be saved.
        output_file (str): Path to the output file.
        output_type (str): Either "csv" or "parquet".
    """
    try:
        if output_type == "parquet":
            df.to_parquet(output_file, index=False)
            logging.info(f"Data successfully saved to Parquet: {output_file}")
        else:  # Default to CSV
            df.to_csv(output_file, index=False)
            logging.info(f"Data successfully saved to CSV: {output_file}")
    except Exception as e:
        logging.error(f"Error saving DataFrame: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Filter factors based on correlation using Dask.")
    parser.add_argument("--file", type=str, required=True, help="Path to the input file (CSV or Parquet).")
    parser.add_argument("--target", type=str, default="mid_price_2", help="Target column name for correlation filtering.")
    parser.add_argument("--corr_ft", type=float, default=0.1, help="Minimum correlation with the target to keep.")
    parser.add_argument("--corr_ff", type=float, default=0.8, help="Maximum correlation between features to keep.")
    parser.add_argument("--output-type", type=str, choices=["csv", "parquet"], default="csv", help="Output file format (default: csv).")
    
    args = parser.parse_args()

    try:
        logging.info("Loading dataset...")

        # Read CSV or Parquet file
        if args.file.endswith(".csv"):
            logging.info("Loading CSV file...")
            df = pd.read_csv(args.file)
        elif args.file.endswith(".parquet"):
            logging.info("Loading Parquet file...")
            df = pd.read_parquet(args.file)
        else:
            raise ValueError("Unsupported file format! Only CSV and Parquet are supported.")

        # Perform filtering using the updated function
        logging.info(f"Filtering features with corr_ft={args.corr_ft} and corr_ff={args.corr_ff}...")
        filtered_df = filter_features_by_correlation_dask(
            df=df,
            target_column=args.target,
            corr_ft=args.corr_ft,
            corr_ff=args.corr_ff,
        )

        # Determine output file extension
        output_file = args.file.replace(".csv", "_filtered").replace(".parquet", "_filtered") + f".{args.output_type}"

        # Save the filtered dataset
        save_to_file(filtered_df, output_file, args.output_type)
        logging.info(f"Filtered dataset saved to: {output_file}")

    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    main()


# Usage Examples:
# 1️⃣ Process a CSV file and output a CSV
#    python filter_factors.py --file mydata.csv --output-type csv
#    # Result: mydata_filtered.csv
#
# 2️⃣ Process a CSV file and output a Parquet
#    python filter_factors.py --file mydata.csv --output-type parquet
#    # Result: mydata.parquet
#
# 3️⃣ Process a Parquet file and output a Parquet
#    python filter_factors.py --file mydata.parquet --output-type parquet
#    # Result: mydata_filtered.parquet
#
# 4️⃣ Process a Parquet file and output a CSV
#    python filter_factors.py --file mydata.parquet --output-type csv
#    # Result: mydata.csv
