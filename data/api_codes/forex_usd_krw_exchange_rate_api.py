# Set the base currency (U.S. dollar) and the target currency (South Korean won)

import requests
import data.config as config

url = "https://api.apilayer.com/exchangerates_data/timeseries?start_date=2022-01-01&end_date=2022-06-30&base=USD&symbols=KRW"

headers = {
    "apikey": config.exchange_rate_api_key
}

response = requests.request("GET", url, headers=headers)
result = response.json()

