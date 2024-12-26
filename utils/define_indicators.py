import pandas as pd
import numpy as np

def calculate_features(df):
    """
    Adds technical indicators and features to the OHLCV dataset.

    Args:
        df (pd.DataFrame): DataFrame with OHLCV data. Columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume'].

    Returns:
        pd.DataFrame: DataFrame with additional features.
    """
    # Moving Averages
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_Middle'] = df['close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Middle'] + (df['close'].rolling(window=20).std() * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['close'].rolling(window=20).std() * 2)

    # Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    # Rate of Change (ROC)
    df['ROC'] = df['close'].pct_change(periods=10) * 100

    # Volume Weighted Average Price (VWAP)
    df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

    # Moving Average Crossovers
    df['SMA_Crossover'] = np.where(df['SMA_20'] > df['SMA_50'], 1, 0)  # 1 for bullish crossover, 0 otherwise

    # Fill missing values
    df.fillna(method='bfill', inplace=True)

    return df
