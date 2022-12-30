import yfinance as yf

# Fetch the data
data = yf.Ticker("AAPL").history(start='2000-01-01', end='2020-12-31')

# Print the data

data.to_csv('data/stocks/AAPL_from_2000_to_2020.csv')