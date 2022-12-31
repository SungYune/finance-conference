"""
Forex Data Collecting API
"""
# https://fred.stlouisfed.org/tags/series?t=&et=&ptic=819202&ob=pv&od=&tg=&tt=
import pandas as pd
from fredapi import Fred
import data.config as config

import warnings

warnings.filterwarnings("ignore")

fred = Fred(api_key=config.fred_api_key)

fund_rate = fred.get_series(
    series_id="DFF",
    observation_start="1980-01-01",
    observation_end="2022-12-31"
)

sp500 = fred.get_series(
    series_id="SP500",
    observation_start="1980-01-01",
    observation_end="2022-12-31"
)

t_note = fred.get_series(
    series_id="T10Y2Y",
    observation_start="1980-01-01",
    observation_end="2022-12-31"
)

dow = fred.get_series(
    series_id="DJIA",
    observation_start="1980-01-01",
    observation_end="2022-12-31"
)

nasdaq = fred.get_series(
    series_id="NASDAQCOM",
    observation_start="1980-01-01",
    observation_end="2022-12-31"
)

exchange_rate = fred.get_series(
    series_id="DEXKOUS",
    observation_start="1980-01-01",
    observation_end="2022-12-31"
)


# Saving Data to csv

def convert_data(df: pd.DataFrame,
                 name: str):
    """Convert Obtained List to DataFrame and Save to CSV"""
    # add a date column using data_frame.index which is a DatetimeIndex
    data_frame = df[:]
    data_frame = data_frame.to_frame()

    data_frame['date'] = pd.to_datetime(data_frame.index)
    data_frame['date'] = data_frame['date'].dt.strftime('%Y-%m-%d')
    data_frame.reset_index(drop=True, inplace=True)

    data_frame.columns = ['value', 'date']
    data_frame = data_frame[['date', 'value']]
    data_frame.to_csv(f'data/forex/{name}.csv')


convert_data(fund_rate, 'fund_rate')
convert_data(sp500, 'sp500')
convert_data(t_note, 't_note')
convert_data(dow, 'dow')
convert_data(nasdaq, 'nasdaq')
convert_data(exchange_rate, 'exchange_rate')
