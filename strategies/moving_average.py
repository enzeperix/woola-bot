import pandas as pd
import numpy as np

def calculate_moving_averages(prices, short_window=40, long_window=100):
    signals = pd.DataFrame(index=prices.index)
    signals['price'] = prices['close']
    signals['short_mavg'] = prices['close'].rolling(window=short_window, min_periods=1).mean()
    signals['long_mavg'] = prices['close'].rolling(window=long_window, min_periods=1).mean()
    signals['signal'] = 0.0
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()
    return signals

