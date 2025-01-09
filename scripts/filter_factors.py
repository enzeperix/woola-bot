import argparse
import logging
from utils.filtering_utils import filter_features_by_correlation_dask

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def save_to_parquet(df, output_file):
    """
    Saves a DataFrame to a Parquet file.
    Args:
        df (pd.DataFrame): DataFrame to be saved.
        output_file (str): Path to the Parquet file.
    Returns:
        None
    """
    try:
        df.to_parquet(output_file, index=False)
        logging.info(f"Data successfully saved to Parquet: {output_file}")
    except Exception as e:
        logging.error(f"Error saving DataFrame to Parquet: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Filter factors based on correlation using Dask.")
    parser.add_argument("--file", type=str, required=True, help="Path to the Parquet file to filter.")
    parser.add_argument("--target", type=str, default="mid_price_2", help="Target column name for correlation filtering.")
    parser.add_argument("--corr_ft", type=float, default=0.1, help="Minimum correlation with the target to keep.")
    parser.add_argument("--corr_ff", type=float, default=0.8, help="Maximum correlation between features to keep.")
    args = parser.parse_args()

    try:
        logging.info("Loading dataset...")
        logging.info("Loading Parquet file...")
        
        # Perform filtering using Dask
        logging.info(f"Filtering features with corr_ft={args.corr_ft} and corr_ff={args.corr_ff}...")
        filtered_df = filter_features_by_correlation_dask(
            parquet_file=args.file,
            target_column=args.target,
            corr_ft=args.corr_ft,
            corr_ff=args.corr_ff
        )

        # Save the filtered dataset
        output_file = args.file.replace(".parquet", "_filtered.parquet")
        save_to_parquet(filtered_df, output_file)
        logging.info(f"Filtered dataset saved to: {output_file}")
    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    main()
