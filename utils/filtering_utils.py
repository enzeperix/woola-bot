import dask.dataframe as dd
import pandas as pd
import numpy as np
import logging

def filter_features_by_correlation_dask(parquet_file, target_column='mid_price_2', corr_ft=0.1, corr_ff=0.8):
    """
    Filters features based on correlation thresholds with the target and other features using Dask.
    Args:
        parquet_file (str): Path to the input Parquet file with features and target column.
        target_column (str): Column to use as the prediction target.
        corr_ft (float): Minimum correlation with the target for a feature to be kept.
        corr_ff (float): Maximum allowed correlation between features.
    Returns:
        pd.DataFrame: Filtered DataFrame with selected features.
    """
    logging.info(f"Filtering features using Dask with corr_ft={corr_ft} and corr_ff={corr_ff}.")

    # Load the Parquet file using Dask
    ddf = dd.read_parquet(parquet_file)
    logging.info("Loaded dataset using Dask.")

    # Separate numeric columns from non-numeric ones
    numeric_df = ddf.select_dtypes(include=[np.number])
    non_numeric_columns = ddf.select_dtypes(exclude=[np.number]).columns

    # Ensure the target column is in the numeric data
    if target_column not in numeric_df.columns:
        raise ValueError(f"Target column '{target_column}' is not numeric or not found in the dataset.")

    # Compute the correlation matrix using Dask
    corr_matrix = numeric_df.corr().compute()
    logging.info("Computed correlation matrix.")

    # Correlation of features with the target column
    target_corr = corr_matrix[target_column].abs().sort_values(ascending=False)

    # Select features based on correlation with target
    selected_features = target_corr[(target_corr >= corr_ft)].index.tolist()
    logging.info(f"Selected {len(selected_features)} features based on target correlation threshold.")

    # Remove the target column from inter-feature correlation filtering
    selected_features.remove(target_column)

    # Filter features to ensure no high inter-feature correlation
    final_features = [target_column]  # Always keep the target column
    for feature in selected_features:
        if all(abs(corr_matrix[feature][f]) <= corr_ff for f in final_features if f in corr_matrix):
            final_features.append(feature)

    logging.info(f"Reduced to {len(final_features)} features after inter-feature correlation filtering.")

    # Combine numeric and non-numeric data back
    filtered_df = ddf[final_features].compute()
    filtered_df = pd.concat([ddf[non_numeric_columns].compute(), filtered_df], axis=1)

    return filtered_df
