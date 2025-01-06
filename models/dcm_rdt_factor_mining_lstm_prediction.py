#!/usr/bin/env python3

import os
import time
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import logging

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# --- Suppress TensorFlow Warnings ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info("Step 0: GPU Configuration: GPU memory growth enabled.")
    except RuntimeError as e:
        logging.error(f"Step 0: GPU Configuration: {e}")

# Save top features to CSV in the same directory as the script
def save_top_features_to_csv(df, input_csv_name):
    """
    Saves the top features DataFrame to a CSV file with a dynamically generated name.
    Args:
        df (pd.DataFrame): The DataFrame containing the top features.
        input_csv_name (str): The name of the input CSV file.
    Returns:
        str: The full path to the saved CSV file.
    """
    base_name = os.path.basename(input_csv_name)
    prefix = os.path.splitext(base_name)[0]
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    output_file_name = f"{prefix}_top_features_{timestamp}.csv"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(script_dir, output_file_name)
    df.to_csv(output_file_path, index=False)
    print(f"Top features saved to: {output_file_path}")
    return output_file_path

# --- Mid-Price Calculation ---
def calculate_mid_prices(df):
    logging.info("Step 1: Mid-Price Calculation: Adding mid-price columns.")
    df['mid_price_1'] = (df['high'] + df['low']) / 2
    df['mid_price_2'] = (df['open'] + df['close']) / 2
    return df

# --- Factor Mining ---
def mine_factors(df, target_column, n_features=100, lags=3):
    """
    Mines top features using Random Forest.
    Args:
        df (pd.DataFrame): DataFrame containing OHLCV and mid-price data.
        target_column (str): Column to use as the prediction target.
        n_features (int): Number of top features to select.
        lags (int): Number of lagged features to generate.
    Returns:
        tuple: (DataFrame with top features, list of top feature names)
    """
    start_time = time.time()
    logging.info("Step 2: Factor Mining: Starting factor mining.")
    df_copy = df.copy()  # Work on a copy of the DataFrame
    # Generate lagged features
    for lag in range(1, lags + 1):
        for col in ['open', 'high', 'low', 'close']:
            df_copy[f'{col}_lag_{lag}'] = df_copy[col].shift(lag)
    df_copy.dropna(inplace=True)  # Remove rows with NaN values after lagging
    logging.info(f"Step 2: Factor Mining: Generated lagged features with {lags} lags.")
    # Prepare data for Random Forest
    X = df_copy[[col for col in df_copy.columns if 'lag' in col]]
    y = np.sign(df_copy[target_column].diff().shift(-1)).replace({0: 1})  # Target: mid-price movement
    # Remove rows with NaN in y
    valid_rows = y.notna()
    X = X[valid_rows]
    y = y[valid_rows]
    logging.info("Step 2: Factor Mining: Prepared data for Random Forest.")
    # Train Random Forest
    rdt = RandomForestClassifier(n_estimators=100, random_state=42)
    rdt.fit(X, y)
    logging.info("Step 2: Factor Mining: Random Forest training completed.")
    # Rank features by importance
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rdt.feature_importances_
    }).sort_values(by='importance', ascending=False)
    top_features = feature_importances.head(n_features)['feature'].tolist()
    # Include mid_price_2 explicitly in the final DataFrame
    top_features_df = df_copy.loc[valid_rows, top_features]
    if 'mid_price_2' in df_copy.columns:
        top_features_df['mid_price_2'] = df_copy.loc[valid_rows, 'mid_price_2']
    # Save top features to CSV
    save_top_features_to_csv(top_features_df, input_csv_name)
    elapsed_time = time.time() - start_time
    logging.info(f"Step 2: Factor Mining: Completed in {elapsed_time:.2f} seconds. Saved top features to CSV.")
    return top_features_df, top_features

# --- LSTM Training ---
def train_lstm(df, top_features, sequence_length=10):
    logging.info("Step 3: LSTM Training: Preparing data and building the model.")
    start_time = time.time()
    if 'mid_price_2' not in df.columns:
        raise ValueError("Column 'mid_price_2' is missing from the DataFrame.")
    for feature in top_features:
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' is missing from the DataFrame.")
    # Prepare data for LSTM
    X_lstm, y_lstm = [], []
    for i in range(len(df) - sequence_length):
        X_seq = df[top_features].iloc[i:i+sequence_length].values
        y_value = np.sign(df['mid_price_2'].diff().shift(-1).iloc[i+sequence_length])
        if not np.isnan(y_value):
            X_lstm.append(X_seq)
            y_lstm.append(y_value)
    X_lstm = np.array(X_lstm)
    y_lstm = np.array(y_lstm)
    # Build LSTM model
    model = Sequential([
        LSTM(50, input_shape=(sequence_length, len(top_features))),
        Dense(1, activation='tanh')
    ])
    model.compile(optimizer='adam', loss='mse')
    # Train the model
    model.fit(X_lstm, y_lstm, epochs=20, batch_size=32)
    elapsed_time = time.time() - start_time
    logging.info(f"Step 3: LSTM Training: Completed in {elapsed_time:.2f} seconds.")
    return model

# --- Evaluation ---
def evaluate_model(model, df, top_features, sequence_length=10):
    logging.info("Step 4: Evaluation: Evaluating the trained model.")
    start_time = time.time()
    X_lstm, y_lstm = [], []
    for i in range(len(df) - sequence_length):
        X_lstm.append(df[top_features].iloc[i:i+sequence_length].values)
        y_lstm.append(np.sign(df['mid_price_2'].diff().shift(-1).iloc[i+sequence_length]))
    X_lstm = np.array(X_lstm)
    y_lstm = np.array(y_lstm)
    y_pred = model.predict(X_lstm)
    y_pred_class = np.where(y_pred > 0, 1, -1)
    cm = confusion_matrix(y_lstm, y_pred_class)
    logging.info(f"Step 4: Evaluation: Confusion Matrix:\n{cm}")
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        raise ValueError("Unexpected confusion matrix shape. Multiclass classification not supported.")
    t = tn + fp + fn + tp
    R_total = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / t
    npbr = ((tp + tn) - R_total) / (t - R_total)
    accuracy = (tp + tn) / t
    elapsed_time = time.time() - start_time
    logging.info(f"Step 4: Evaluation: Completed in {elapsed_time:.2f} seconds.")
    return {'accuracy': accuracy, 'npbr': npbr}

# --- Main Pipeline ---
if __name__ == "__main__":
    # Parse command-line arguments for input CSV
    parser = argparse.ArgumentParser(description="LSTM Model for Factor Mining and Prediction")
    parser.add_argument("input_csv", help="Path to the input CSV file containing OHLCV data")
    args = parser.parse_args()

    # Extract the input file name
    input_csv_name = args.input_csv

    # Start logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Step 0: Starting the pipeline.")

    # Step 0: Load the dataset
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    try:
        df = pd.read_csv(input_csv_name, names=columns)
        logging.info(f"Step 0: Loaded DataFrame with columns: {df.columns}")
    except Exception as e:
        logging.error(f"Step 0: Failed to load input CSV. Error: {e}")
        exit(1)

    # Step 1: Add mid-prices
    try:
        df = calculate_mid_prices(df)
        logging.info("Step 1: Mid-Price Calculation: Completed.")
    except Exception as e:
        logging.error(f"Step 1: Mid-Price Calculation failed. Error: {e}")
        exit(1)

    # Step 2: Mine factors
    try:
        top_features_df, top_features = mine_factors(df, target_column='mid_price_2', n_features=100)
        logging.info(f"Step 2: Factor Mining: Completed. Top features identified: {top_features[:5]}...")
        # Save the top features to a CSV
        save_top_features_to_csv(top_features_df, input_csv_name)
    except Exception as e:
        logging.error(f"Step 2: Factor Mining failed. Error: {e}")
        exit(1)

    # Step 3: Train LSTM
    try:
        model = train_lstm(top_features_df, top_features)
        logging.info("Step 3: LSTM Training: Completed.")
    except Exception as e:
        logging.error(f"Step 3: LSTM Training failed. Error: {e}")
        exit(1)

    # Step 4: Evaluate the model
    try:
        metrics = evaluate_model(model, top_features_df, top_features)
        logging.info(f"Step 4: Evaluation: Completed. Accuracy = {metrics['accuracy']:.2%}, NPBR = {metrics['npbr']:.2%}")
    except Exception as e:
        logging.error(f"Step 4: Evaluation failed. Error: {e}")
        exit(1)

    logging.info("Pipeline completed successfully.")



# If you want to expand on this, here are some ideas:
# - At target creation step. Choose other target than the mid_price_2. For example create a mid_price_3 column and use it as the target.
#   mid_price_3 = (mid_price_1 + mid_price_2) / 2
#   Inspect the Mid-Price Columns:
#   Visualize mid_price_1 and mid_price_2 over time to confirm they provide meaningful trends.
# - At factor mining step you can experiment with different values for n_estimators (number of decision trees) and random_state.
#   Also try different values for n_features and lags.
# 
#
