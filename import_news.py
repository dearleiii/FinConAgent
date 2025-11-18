import requests

API_KEY = 'PKR4CLSTT64VDIHC47P8'
SECRET_KEY = 'aiBx2WHPUP3WES8jhdJ9ClpxmFpFH9ZMRCyU8AXz'

BASE_URL = 'https://data.alpaca.markets/v1beta1/news'  # For News API (v1beta1)

params = {
    'symbols': 'AAPL,MSFT',
    'limit': 5,
    'start': '2023-09-01T00:00:00Z'
}

response = requests.get(
    BASE_URL,
    headers={
        'APCA-API-KEY-ID': API_KEY,
        'APCA-API-SECRET-KEY': SECRET_KEY
    },
    params=params
)

if response.status_code == 200:
    news_data = response.json()
    for article in news_data['news']:
        print(f"{article['headline']} - {article['source']}")
        print(article['summary'])
        print()
else:
    print(f"Error {response.status_code}: {response.text}")