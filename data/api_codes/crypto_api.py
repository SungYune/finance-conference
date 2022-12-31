"""
Crypto Data Collecting API
"""

from binance import Client
import pandas as pd

import data.config as config

# Set the API key and secret
client = Client(api_key=config.binance_public,
                api_secret=config.binance_secret)

# Get the price of Bitcoin
btc_data_5min  = client.get_historical_klines(
    symbol="BTCUSDP",
    interval=Client.KLINE_INTERVAL_5MINUTE,
    start_str="1 Jan, 2017",
    end_str="1 Jan, 2022"
)

btc_data_15min = client.get_historical_klines(
    symbol="BTCUSDP",
    interval=Client.KLINE_INTERVAL_15MINUTE,
    start_str="1 Jan, 2017",
    end_str="1 Jan, 2022"
)


# Save it to csv
def convert_data(data: list, interval: str) -> pd.DataFrame:
    """Convert Obtained List to DataFrame and Save to CSV"""

    columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume',
               'Close time', 'Quote asset volume', 'Number of trades',
               'Taker buy base asset volume', 'Taker buy quote asset volume',
               'Ignore']

    df = pd.DataFrame(data, columns=columns)

    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')
    df = df.drop(columns=['Ignore'])

    df.to_csv(f'data/crypto/btc_data_{interval}.csv')
    return df


btc_data_5min  = convert_data(btc_data_5min,  '5min')
btc_data_15min = convert_data(btc_data_15min, '15min')
