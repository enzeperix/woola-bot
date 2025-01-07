def generate_expanded_differences(df, lags=[1, 2, 3, 5, 8]):
    """
    Generates a comprehensive set of differences and lagged factors for price components and volume.
    Args:
        df (pd.DataFrame): DataFrame containing OHLCV data.
        lags (list): List of lag values for generating lagged differences.
        With 5 lags the function will generate a total of 92 features ( 2 static differences + 90 lagged differences).
    Returns:
        pd.DataFrame: DataFrame with additional expanded difference factors.
    """
    # Static factors (immediate differences within the same row)
    df['high_minus_low'] = df['high'] - df['low']
    df['close_minus_open'] = df['close'] - df['open']
    # Lagged differences for price components and volume
    for lag in lags:
        # Price component differences
        df[f'high_minus_high_lag_{lag}'] = df['high'] - df['high'].shift(lag)
        df[f'low_minus_low_lag_{lag}'] = df['low'] - df['low'].shift(lag)
        df[f'open_minus_open_lag_{lag}'] = df['open'] - df['open'].shift(lag)
        df[f'close_minus_close_lag_{lag}'] = df['close'] - df['close'].shift(lag)
        # Volume differences
        df[f'volume_minus_volume_lag_{lag}'] = df['volume'] - df['volume'].shift(lag)
        # Differences between price components across time
        df[f'high_minus_close_lag_{lag}'] = df['high'] - df['close'].shift(lag)
        df[f'low_minus_open_lag_{lag}'] = df['low'] - df['open'].shift(lag)
        df[f'close_minus_high_lag_{lag}'] = df['close'] - df['high'].shift(lag)
        df[f'open_minus_low_lag_{lag}'] = df['open'] - df['low'].shift(lag)
        # Absolute differences (magnitude-only changes)
        df[f'abs_high_minus_high_lag_{lag}'] = abs(df['high'] - df['high'].shift(lag))
        df[f'abs_close_minus_close_lag_{lag}'] = abs(df['close'] - df['close'].shift(lag))
        # Differences of differences (acceleration or deceleration)
        df[f'diff_of_diff_close_lag_{lag}'] = df['close'].diff().diff(periods=lag)
        df[f'diff_of_diff_volume_lag_{lag}'] = df['volume'].diff().diff(periods=lag)
        # Volume-normalized differences
        df[f'close_diff_normalized_lag_{lag}'] = np.where(df['volume'] != 0,
                                                          (df['close'] - df['close'].shift(lag)) / df['volume'], 0)
        # Range-based differences
        df[f'close_minus_range_lag_{lag}'] = df['close'] - (df['high'] - df['low']).shift(lag)
        # Directional indicators
        df[f'direction_close_lag_{lag}'] = np.sign(df['close'] - df['close'].shift(lag))
        df[f'direction_high_lag_{lag}'] = np.sign(df['high'] - df['high'].shift(lag))
        df[f'direction_volume_lag_{lag}'] = np.sign(df['volume'] - df['volume'].shift(lag))
        # Composite differences
        df[f'composite_high_low_diff_lag_{lag}'] = (df['high'] - df['low']) - (df['high'].shift(lag) - df['low'].shift(lag))
        df[f'composite_open_close_diff_lag_{lag}'] = (df['open'] - df['close']) - (df['open'].shift(lag) - df['close'].shift(lag))
    # Handle missing values (introduced by lagging)
    df.fillna(0, inplace=True)

    return df


def generate_ratios(df, lags=[1, 2, 3, 5, 8], imputation_method='default'):
    """
    Generates ratio features for price components, volume, and derived metrics.
    Args:
        df (pd.DataFrame): DataFrame containing OHLCV data.
        lags (list): List of lag values for generating lagged ratios.
        imputation_method (str): Method for handling missing values ('default', 'ffill').
        With 5 lags, the function will generate 108 ratio-related features in total ( 18 original ratios + 90 lagged ratios).
    Returns:
        pd.DataFrame: DataFrame with additional ratio features.
    """
    # --- Basic Ratios ---
    df['high_low_ratio'] = np.where(df['low'] != 0, df['high'] / df['low'], 1)
    df['close_open_ratio'] = np.where(df['open'] != 0, df['close'] / df['open'], 1)
    df['high_close_ratio'] = np.where(df['close'] != 0, df['high'] / df['close'], 1)
    df['low_close_ratio'] = np.where(df['close'] != 0, df['low'] / df['close'], 1)
    df['open_close_ratio'] = np.where(df['close'] != 0, df['open'] / df['close'], 1)
    # --- Range Ratios ---
    df['range_ratio'] = np.where(df['close'] != 0, (df['high'] - df['low']) / df['close'], 0)
    df['volume_range_ratio'] = np.where((df['high'] - df['low']) != 0, df['volume'] / (df['high'] - df['low']), 0)
    df['average_price_ratio'] = np.where(df['close'] != 0, (df['open'] + df['high'] + df['low'] + df['close']) / 4 / df['close'], 1)
    # --- Enhanced Ratios ---
    # Mid-Price Ratios
    df['mid_price_to_close_ratio'] = np.where(df['close'] != 0, ((df['high'] + df['low']) / 2) / df['close'], 1)
    df['mid_price_to_range_ratio'] = np.where((df['high'] - df['low']) != 0, ((df['high'] + df['low']) / 2) / (df['high'] - df['low']), 0) 
    # Range-Adjusted Ratios
    df['adjusted_high_low_ratio'] = np.where((df['high'] + df['low']) != 0, (df['high'] - df['low']) / (df['high'] + df['low']), 0)
    # Volatility Ratios
    df['range_to_avg_price_ratio'] = np.where(((df['high'] + df['low'] + df['open'] + df['close']) / 4) != 0, 
                                              (df['high'] - df['low']) / ((df['high'] + df['low'] + df['open'] + df['close']) / 4), 
                                              0)
    df['close_fluctuation_ratio'] = np.where(df['close'].shift(1) != 0, 
                                             abs(df['close'] - df['close'].shift(1)) / df['close'].shift(1), 
                                             0)
    df['open_fluctuation_ratio'] = np.where(df['open'].shift(1) != 0, 
                                            abs(df['open'] - df['open'].shift(1)) / df['open'].shift(1), 
                                            0)
    # Volume Ratios
    df['volume_to_avg_price_ratio'] = np.where(((df['high'] + df['low'] + df['open'] + df['close']) / 4) != 0, 
                                               df['volume'] / ((df['high'] + df['low'] + df['open'] + df['close']) / 4), 
                                               0)
    df['volume_to_mid_price_ratio'] = np.where(((df['high'] + df['low']) / 2) != 0, df['volume'] / ((df['high'] + df['low']) / 2), 0)
    df['volume_change_ratio'] = np.where(df['volume'].shift(1) != 0, abs(df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1), 0)
    # Momentum Ratios
    df['momentum_high_low_ratio'] = np.where(df['high'].shift(1) != 0, (df['high'] - df['low']) / df['high'].shift(1), 0)
    df['momentum_close_open_ratio'] = np.where(df['open'].shift(1) != 0, (df['close'] - df['open']) / df['open'].shift(1), 0)
    # --- Lagged Ratios ---
    for lag in lags:
        for col in df.columns:
            if '_ratio' in col:  # Add lagged versions for all ratio columns
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    # --- Handle Missing Values ---
    if imputation_method == 'ffill':
        df.fillna(method='ffill', inplace=True)
    else:  # Default to neutral values
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)

    return df


def generate_pairwise_operations(df, lags=[1, 2, 3, 5, 8], rolling_windows=[3, 5, 8], imputation_method='default'):
    """
    Generates features based on pairwise operations for price components, mid-price analysis,
    and includes lagged and rolling statistics for specified lags and rolling windows.
    Args:
        df (pd.DataFrame): DataFrame containing OHLCV data.
        lags (list): List of lag values for generating lagged features.
        rolling_windows (list): List of window sizes for rolling aggregations.
        imputation_method (str): Method for handling missing values ('default', 'ffill', 'interpolate').
        With 5 lags and 3 rolling windows, the function will generate a significantly expanded feature set:
        Total Features: 24 + 24x5 + 9x2x3 = 24 + 120 + 54 = 198.
    Returns:
        pd.DataFrame: DataFrame with additional pairwise operation features, their lags, and rolling statistics.
    """
    # --- Price Gaps ---
    df['open_to_high_gap'] = df['high'] - df['open']
    df['close_to_low_gap'] = df['close'] - df['low']
    df['open_to_low_gap'] = df['open'] - df['low']
    df['close_to_high_gap'] = df['high'] - df['close']

    # --- Mid-Price Analysis ---
    df['mid_price_1'] = (df['high'] + df['low']) / 2
    df['mid_price_2'] = (df['open'] + df['close']) / 2
    df['mid_price_difference'] = df['mid_price_1'] - df['mid_price_2']

    # --- Skewness in Price Movement ---
    df['high_minus_mid_price_1'] = df['high'] - df['mid_price_1']
    df['low_minus_mid_price_1'] = df['low'] - df['mid_price_1']
    df['close_minus_mid_price_2'] = df['close'] - df['mid_price_2']
    df['open_minus_mid_price_2'] = df['open'] - df['mid_price_2']

    # --- Average Price Analysis ---
    df['average_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['average_price_to_close_gap'] = df['average_price'] - df['close']
    df['average_price_to_open_gap'] = df['average_price'] - df['open']

    # --- Range-Based Features ---
    df['price_range'] = df['high'] - df['low']
    df['range_to_open_gap'] = df['price_range'] - (df['open'] - df['low'])
    df['range_to_close_gap'] = df['price_range'] - (df['high'] - df['close'])

    # --- Directional Skewness ---
    df['directional_skew_open'] = np.where(df['price_range'] != 0, (df['open'] - df['low']) / df['price_range'], 0)
    df['directional_skew_close'] = np.where(df['price_range'] != 0, (df['close'] - df['low']) / df['price_range'], 0)

    # --- Relative Strength Indicators ---
    df['strength_open_vs_close'] = np.where(df['close'] != 0, df['open'] / df['close'], 1)
    df['strength_high_vs_low'] = np.where(df['low'] != 0, df['high'] / df['low'], 1)

    # --- Additional Pairwise Features ---
    df['volume_to_range_ratio'] = np.where(df['price_range'] != 0, df['volume'] / df['price_range'], 0)
    df['range_to_mid_price_gap'] = df['price_range'] - df['mid_price_1']
    df['mid_price_to_average_price_gap'] = df['mid_price_1'] - df['average_price']

    # --- Lagged Features ---
    for lag in lags:
        for col in [
            'open_to_high_gap', 'close_to_low_gap', 'open_to_low_gap', 'close_to_high_gap',
            'mid_price_1', 'mid_price_2', 'mid_price_difference',
            'high_minus_mid_price_1', 'low_minus_mid_price_1',
            'close_minus_mid_price_2', 'open_minus_mid_price_2',
            'average_price', 'average_price_to_close_gap', 'average_price_to_open_gap',
            'price_range', 'range_to_open_gap', 'range_to_close_gap',
            'directional_skew_open', 'directional_skew_close',
            'strength_open_vs_close', 'strength_high_vs_low',
            'volume_to_range_ratio', 'range_to_mid_price_gap', 'mid_price_to_average_price_gap'
        ]:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # --- Rolling Aggregations ---
    for window in rolling_windows:
        for col in [
            'open_to_high_gap', 'close_to_low_gap', 'mid_price_difference',
            'high_minus_mid_price_1', 'low_minus_mid_price_1',
            'average_price', 'price_range', 'directional_skew_open', 'strength_open_vs_close'
        ]:
            df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
            df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()

    # --- Handle Missing Values ---
    if imputation_method == 'ffill':
        df.fillna(method='ffill', inplace=True)
    elif imputation_method == 'interpolate':
        df.interpolate(method='linear', inplace=True)
    else:  # Default to neutral values
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)

    return df


def generate_rolling_statistics(df, windows=[3, 5, 8, 13, 21], imputation_method='default'):
    """
    Generates rolling statistics for price components and volume using Fibonacci sequence for window sizes.
    Args:
        df (pd.DataFrame): DataFrame containing OHLCV data.
        windows (list): List of window sizes for calculating rolling statistics (default: Fibonacci sequence).
        imputation_method (str): Method for handling missing values ('default', 'ffill', 'interpolate').
        For 5 windows we have: Per Window: 10 features for means, 10 for standard deviations, 4 for ranges = 24 features/window. Total: 24x5 = 120 features.
    Returns:
        pd.DataFrame: DataFrame with additional rolling statistic features.
    """
    # --- Rolling Mean (Price Trends) ---
    for window in windows:
        df[f'rolling_mean_close_{window}'] = df['close'].rolling(window).mean()
        df[f'rolling_mean_open_{window}'] = df['open'].rolling(window).mean()
        df[f'rolling_mean_high_{window}'] = df['high'].rolling(window).mean()
        df[f'rolling_mean_low_{window}'] = df['low'].rolling(window).mean()
        df[f'rolling_mean_volume_{window}'] = df['volume'].rolling(window).mean()

    # --- Rolling Standard Deviation (Volatility) ---
    for window in windows:
        df[f'rolling_std_close_{window}'] = df['close'].rolling(window).std()
        df[f'rolling_std_open_{window}'] = df['open'].rolling(window).std()
        df[f'rolling_std_high_{window}'] = df['high'].rolling(window).std()
        df[f'rolling_std_low_{window}'] = df['low'].rolling(window).std()
        df[f'rolling_std_volume_{window}'] = df['volume'].rolling(window).std()

    # --- Rolling Ranges ---
    for window in windows:
        df[f'rolling_range_mean_{window}'] = (df['high'] - df['low']).rolling(window).mean()
        df[f'rolling_range_std_{window}'] = (df['high'] - df['low']).rolling(window).std()
        df[f'rolling_range_max_{window}'] = (df['high'] - df['low']).rolling(window).max()
        df[f'rolling_range_min_{window}'] = (df['high'] - df['low']).rolling(window).min()

    # --- Handle Missing Values ---
    if imputation_method == 'ffill':
        df.fillna(method='ffill', inplace=True)
    elif imputation_method == 'interpolate':
        df.interpolate(method='linear', inplace=True)
    else:  # Default to neutral values
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)

    return df


# Refined Recommendations for generate_simple_differences and generate_ratios:
# 1. Lagged Factors and Simple Differences
# Neutral Placeholder (0): For lagged values or differences, use 0. This ensures no artificial momentum is introduced, and computational performance is optimal.
# 2. Ratios
# Mixed Approach:
# Use 1 as the default for basic ratios like close/open or high/low. This keeps ratios neutral and avoids division errors.
# For lagged ratios, forward fill (ffill) is better since it preserves temporal continuity in trends.
# 3. Rolling Statistics
# Forward Fill (ffill):
# Rolling windows often produce consecutive missing values (e.g., the first n rows for a window of size n).
# Forward fill is a logical choice for rolling averages, as it ensures the series remains smooth without introducing abrupt changes.
# 4. Outliers and Infinities
# Replace with 0:
# Any infinities (inf, -inf) should be replaced with 0 to prevent issues during training or analysis.
# Final Thoughts:
# With large datasets, computational efficiency is key, so default neutral values (0 or 1) are often the best starting point.
# If you notice artifacts in certain features, try forward fill or interpolation selectively.
# Test your model's performance to validate that these imputation choices don’t harm predictive accuracy.


# !!!!!!!!!!!!! Experiment with different imputation methods and see how they affect model performance. !!!!!!!!!!!!!
# ok so when i get to the training part of the model i have to keep in mind to experiment with both default and ffill
# ChatGPT said:
# ChatGPT
# Exactly! Experimenting with both default imputation (using 0) and forward-fill (ffill) during the training phase is a key step in ensuring the best model performance. Here's a practical way to think about it:
# 
# Why Experiment?
# Default (0):
# 
# Treats missing values as neutral.
# Suitable for ensuring no artificial trends are introduced.
# May lead to slightly weaker performance if the model heavily relies on lagged features.
# Forward-Fill (ffill):
# 
# Assumes continuity in time series data.
# Preserves trends and patterns in lagged features.
# May perform better in cases where recent values strongly influence predictions.
# Impact of Choice:
# 
# The optimal imputation method often depends on the dataset and the behavior of the target variable.
# For example, in financial data, momentum is often critical, so ffill may align better with observed trends.
# How to Experiment?
# 1. Modify the Imputation Step:
# In the mine_factors function or any feature-engineering step:
# 
# python
# Copy code
# # Default Imputation
# df.fillna(0, inplace=True)
# 
# # Forward-Fill Imputation (alternative)
# df.fillna(method='ffill', inplace=True)
# 2. Split Training and Testing:
# Train the model with features generated using both imputation methods:
# 
# Model 1: Features imputed with 0.
# Model 2: Features imputed with ffill.
# 3. Compare Results:
# Evaluate each model on:
# 
# Accuracy
# NPBR (Net Profit Bias Ratio) or other domain-specific metrics.
# 4. Log and Document:
# Keep track of:
# 
# Which imputation method performs better.
# Whether certain datasets favor one method over the other.
#-----------------------------------------------------------
# 1. Feature Normalization
# Why: Some pairwise features might produce very large or very small values (e.g., differences between high and low prices), which can dominate the learning process.
# How: Add optional normalization or scaling to the pairwise features, such as:
# Scaling by dividing by the range or a base price (high - low, close, etc.).
# Standardizing to mean 0 and standard deviation 1.
# Example:
# 
# python
# Copy code
#  df['normalized_open_to_high_gap'] = df['open_to_high_gap'] / (df['high'] - df['low'])
# 
# Recommendation for Your Case
# Given your plan:
# 
# Feature Filtering: No normalization needed during this step.
# Model Training: Normalize the features after filtering. Use techniques like:
# Z-score normalization: For LSTMs and models that benefit from zero-mean, unit-variance input.
# Min-max scaling: For cases where you want to scale features to a specific range (e.g., [0, 1]).
# This approach balances the computational overhead of normalization with its benefits during training.