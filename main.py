import ccxt
import pandas as pd
from strategies.moving_average import calculate_moving_averages

def main():
    # Initialize exchange
    exchange = ccxt.binance()
    
    # Fetch market data
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1d')
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Calculate signals
    signals = calculate_moving_averages(df)
    
    # Print signals for now (later, this will be trading logic)
    print(signals)

if __name__ == "__main__":
    main()

