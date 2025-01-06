import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class OHLCModel:
    """
    A class to implement feature extraction, HMM model training, and predictions
    based on OHLCV data.
    """

    def __init__(self, n_states=2):
        """
        Initialize the model.

        Args:
            n_states (int): Number of hidden states for the HMM model.
        """
        self.n_states = n_states
        self.hmm_model = None
        self.top_features = None
        self.lstm_model = None

    def preprocess_data(self, file_path, lags=3):
        """
        Load and preprocess OHLCV data, including generating lagged features.

        Args:
            file_path (str): Path to the CSV file containing OHLCV data.
            lags (int): Number of lagged features to generate.

        Returns:
            pd.DataFrame: Preprocessed data.
        """
        df = pd.read_csv(file_path, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['mid_price'] = (df['open'] + df['close']) / 2  # Mid-price

        # Generate lagged features
        for lag in range(1, lags + 1):
            for col in ['open', 'high', 'low', 'close', 'mid_price']:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

        # Drop NaN rows after generating lags
        df.dropna(inplace=True)
        return df

    def factor_mining(self, df):
        """
        Perform feature selection using Random Forest and select top features.

        Args:
            df (pd.DataFrame): DataFrame with preprocessed data.

        Returns:
            list: Top feature names.
        """
        X = df[[col for col in df.columns if 'lag' in col]]
        y = np.sign(df['mid_price'].diff().shift(-1)).replace({0: 1})

        # Train Random Forest
        rdt = RandomForestClassifier(n_estimators=100)
        rdt.fit(X, y)

        # Rank features by importance
        feature_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': rdt.feature_importances_
        }).sort_values(by='importance', ascending=False)

        self.top_features = feature_importances.head(100)['feature'].tolist()
        return self.top_features

    def train_hmm(self, df):
        """
        Train a Hidden Markov Model (HMM) on the selected features.

        Args:
            df (pd.DataFrame): DataFrame with preprocessed data.
        """
        data = df[self.top_features].dropna()
        self.hmm_model = GaussianHMM(n_components=self.n_states, covariance_type="diag", n_iter=1000)
        self.hmm_model.fit(data)

        # Add hidden states to the DataFrame
        df['hidden_state'] = self.hmm_model.predict(data)
        return df

    def train_lstm(self, df, sequence_length=10):
        """
        Train an LSTM model on the selected features.

        Args:
            df (pd.DataFrame): DataFrame with preprocessed data.
            sequence_length (int): Number of time steps for LSTM input.

        Returns:
            None
        """
        data = df[self.top_features]
        target = np.sign(df['mid_price'].diff().shift(-1)).replace({0: 1})

        # Prepare data for LSTM
        X_lstm, y_lstm = [], []
        for i in range(len(data) - sequence_length):
            X_lstm.append(data.iloc[i:i+sequence_length].values)
            y_lstm.append(target.iloc[i+sequence_length])

        X_lstm = np.array(X_lstm)
        y_lstm = np.array(y_lstm)

        # Define LSTM model
        self.lstm_model = Sequential([
            LSTM(50, input_shape=(sequence_length, len(self.top_features))),
            Dense(1, activation='tanh')
        ])
        self.lstm_model.compile(optimizer='adam', loss='mse')

        # Train the model
        self.lstm_model.fit(X_lstm, y_lstm, epochs=20, batch_size=32)

    def predict(self, df):
        """
        Make predictions using the trained LSTM model.

        Args:
            df (pd.DataFrame): DataFrame with preprocessed data.

        Returns:
            np.array: Predicted values.
        """
        data = df[self.top_features]
        X_lstm = []
        for i in range(len(data) - 10):
            X_lstm.append(data.iloc[i:i+10].values)

        X_lstm = np.array(X_lstm)
        predictions = self.lstm_model.predict(X_lstm)
        return predictions

    def save_states(self, df, output_file):
        """
        Save the DataFrame with hidden states to a CSV file.

        Args:
            df (pd.DataFrame): DataFrame with hidden states added.
            output_file (str): Path to save the output file.
        """
        df.to_csv(output_file, index=False)
        print(f"Saved hidden states to {output_file}.")



# How to Use the File
# Create the Model Instance:
# 
# from models.hmm_model import OHLCModel
# 
# model = OHLCModel(n_states=3)  # Initialize with 3 hidden states
# Preprocess Data:
# 
# df = model.preprocess_data("data/bybit_data/SOLUSDT/consolidated_1m-timeframe.csv")
# Mine Factors:
# 
# top_features = model.factor_mining(df)
# print("Top features:", top_features)
# Train HMM:
# 
# df_with_states = model.train_hmm(df)
# model.save_states(df_with_states, "data/bybit_data/SOLUSDT/1m_with_hmm.csv")
# Train LSTM:
# 
# model.train_lstm(df)
# Make Predictions:
# 
# predictions = model.predict(df)
# print("Predictions:", predictions)