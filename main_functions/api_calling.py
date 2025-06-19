import pandas as pd
import requests

def get_last_100_day_data(ticker):
    api_key = '8U4V3SZWG1XYKDJM'
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}'
    r = requests.get(url)
    data = r.json()
    
    if 'Time Series (Daily)' not in data:
        raise ValueError(f"Could not get daily time series data for {ticker}. Response: {data}")
        
    time_series = data['Time Series (Daily)']
    filtered_data = []
    
    for date in time_series:
        filtered_data.append({
            'date': date,
            'close': float(time_series[date]['4. close']),
            'volume': int(time_series[date]['5. volume'])
        })
            
    return pd.DataFrame(filtered_data).set_index('date')

def get_monthly_data(ticker):
    api_key = '8U4V3SZWG1XYKDJM'
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={ticker}&apikey={api_key}&outputsize=full'
    r = requests.get(url)
    data = r.json()
    
    if 'Monthly Time Series' not in data:
        raise ValueError(f"Could not get monthly time series data for {ticker}. Response: {data}")
        
    time_series = data['Monthly Time Series']
    filtered_data = []
    
    for date in time_series:
        filtered_data.append({
            'date': date,
            'close': float(time_series[date]['4. close']),
            'volume': int(time_series[date]['5. volume'])
        })
            
    return pd.DataFrame(filtered_data).set_index('date')