import requests
import pandas as pd

# Set the API key
api_key = 'YOUR_API_KEY'

# Set the ticker for the CPI data series
ticker = 'CPALTT01USM657N'

# Set the start and end dates for the data
start_date = '1970-01-01'
end_date = '2022-12-31'

# Make a request to the FRED API to retrieve the data
response = requests.get(f'https://api.stlouisfed.org/fred/series/observations?series_id={ticker}&observation_start={start_date}&observation_end={end_date}&api_key={api_key}')

# Convert the response to a pandas dataframe
df = pd.DataFrame.from_dict(response.json()['observations'])

# Display the data
print(df)
