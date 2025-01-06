def generate_simple_differences(df, lags=[1, 2, 3, 5, 8]):
    """
    Generates simple differences and lagged factors for price components and volume.
    Args:
        df (pd.DataFrame): DataFrame containing OHLCV data.
        lags (list): List of lag values for generating lagged differences. Lags values are picked in Fibonacci sequence.
    Returns:
        pd.DataFrame: DataFrame with additional simple difference factors.
        For 5 lags the function will generate a total of 25 + 2 = 27 features.
    """
    # Static factors
    df['high_minus_low'] = df['high'] - df['low']
    df['close_minus_open'] = df['close'] - df['open']

    # Lagged differences for price components
    for lag in lags:
        df[f'high_minus_high_lag_{lag}'] = df['high'] - df['high'].shift(lag)
        df[f'low_minus_low_lag_{lag}'] = df['low'] - df['low'].shift(lag)
        df[f'open_minus_open_lag_{lag}'] = df['open'] - df['open'].shift(lag)
        df[f'close_minus_close_lag_{lag}'] = df['close'] - df['close'].shift(lag)
        df[f'volume_minus_volume_lag_{lag}'] = df['volume'] - df['volume'].shift(lag)

    # Drop rows with NaN introduced by lagging
    df.dropna(inplace=True)

    return df


def generate_ratios(df, lags=[1, 2, 3, 5, 8]):
    """
    Generates ratio features for price components, volume, and derived metrics.
    Args:
        df (pd.DataFrame): DataFrame containing OHLCV data.
        lags (list): List of lag values for generating lagged ratios.
    Returns:
        pd.DataFrame: DataFrame with additional ratio features.
        For 5 lags the function will generate 8 * 5 = 40 ratios based on lagged factors + 8 basic ratios = 48 features.
    """
    # --- Basic Ratios ---
    # Relative Price Levels
    df['high_low_ratio'] = np.where(df['low'] != 0, df['high'] / df['low'], 0)
    df['close_open_ratio'] = np.where(df['open'] != 0, df['close'] / df['open'], 0)
    df['high_close_ratio'] = np.where(df['close'] != 0, df['high'] / df['close'], 0)
    df['low_close_ratio'] = np.where(df['close'] != 0, df['low'] / df['close'], 0)
    df['open_close_ratio'] = np.where(df['close'] != 0, df['open'] / df['close'], 0)

    # --- Range Ratios ---
    df['range_ratio'] = np.where(df['close'] != 0, (df['high'] - df['low']) / df['close'], 0)

    # --- Volume Ratios ---
    df['volume_range_ratio'] = np.where((df['high'] - df['low']) != 0, df['volume'] / (df['high'] - df['low']), 0)
    df['average_price_ratio'] = np.where(df['close'] != 0, (df['open'] + df['high'] + df['low'] + df['close']) / 4 / df['close'], 0)

    # Replace any inf or NaN values with 0
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    # --- Lagged Ratios ---
    for lag in lags:
        for col in ['high_low_ratio', 'close_open_ratio', 'high_close_ratio',
                    'low_close_ratio', 'open_close_ratio', 'range_ratio',
                    'volume_range_ratio', 'average_price_ratio']:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # Drop NaN values after adding lagged features
    df.fillna(0, inplace=True)

    return df

