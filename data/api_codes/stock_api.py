"""
Stock Data Collecting API
"""

import yfinance as yf

# Fetch the data
aapl = yf.Ticker("AAPL").history(start='2000-01-01', end='2022-12-31')
amzn = yf.Ticker("AMZN").history(start='2000-01-01', end='2022-12-31')
meta = yf.Ticker("META").history(start='2000-01-01', end='2022-12-31')
goog = yf.Ticker("GOOG").history(start='2000-01-01', end='2022-12-31')
msft = yf.Ticker("MSFT").history(start='2000-01-01', end='2022-12-31')

tsla = yf.Ticker("TSLA").history(start='2000-01-01', end='2022-12-31')
nflx = yf.Ticker("NFLX").history(start='2000-01-01', end='2022-12-31')

# Saving Data to csv
aapl.to_csv("data/stocks/aapl.csv")
amzn.to_csv("data/stocks/amzn.csv")
meta.to_csv("data/stocks/meta.csv")
goog.to_csv("data/stocks/goog.csv")
msft.to_csv("data/stocks/msft.csv")

tsla.to_csv("data/stocks/tsla.csv")
nflx.to_csv("data/stocks/nflx.csv")
