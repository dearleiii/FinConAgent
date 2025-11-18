from datetime import datetime
import time
from pandas.tseries.offsets import BDay  # BDay is business day, not birthday...

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame


# DATA_API_KEY = args.data_key
# DATA_API_SECRET = args.data_secret
DATA_API_BASE_URL = "wss://data.alpaca.markets"
TRADING_API_KEY = "PKR4CLSTT64VDIHC47P8"
TRADING_API_SECRET = "aiBx2WHPUP3WES8jhdJ9ClpxmFpFH9ZMRCyU8AXz"
TRADING_API_BASE_URL = "https://paper-api.alpaca.markets"

today = datetime.today()

TEST_END_DATE = (today - BDay(0)).to_pydatetime().date()
TEST_START_DATE = (TEST_END_DATE - BDay(1)).to_pydatetime().date()
TRAIN_END_DATE = (TEST_START_DATE - BDay(1)).to_pydatetime().date()
TRAIN_START_DATE = (TRAIN_END_DATE - BDay(5)).to_pydatetime().date()
TRAINFULL_START_DATE = TRAIN_START_DATE
TRAINFULL_END_DATE = TEST_END_DATE

TRAIN_START_DATE = str(TRAIN_START_DATE)
TRAIN_END_DATE = str(TRAIN_END_DATE)
TEST_START_DATE = str(TEST_START_DATE)
TEST_END_DATE = str(TEST_END_DATE)
TRAINFULL_START_DATE = str(TRAINFULL_START_DATE)
TRAINFULL_END_DATE = str(TRAINFULL_END_DATE)

print("TRAIN_START_DATE: ", TRAIN_START_DATE)
print("TRAIN_END_DATE: ", TRAIN_END_DATE)
print("TEST_START_DATE: ", TEST_START_DATE)
print("TEST_END_DATE: ", TEST_END_DATE)
print("TRAINFULL_START_DATE: ", TRAINFULL_START_DATE)
print("TRAINFULL_END_DATE: ", TRAINFULL_END_DATE)

# Initialize the TradingClient for paper trading
trading_client = TradingClient(TRADING_API_KEY, TRADING_API_SECRET, paper=True)

# Example: Get account information
account = trading_client.get_account()
print(account)


stock_data_client = StockHistoricalDataClient(TRADING_API_KEY, TRADING_API_SECRET)
# Request historical bars for a stock
request_params = StockBarsRequest(
    symbol_or_symbols=["AAPL"],
    timeframe=TimeFrame.Day,
    start=datetime(2025, 10, 5),
    end=datetime(2025, 10, 9)
)

bars = stock_data_client.get_stock_bars(request_params)
print("\n\n check stock .....")
print(bars)


### Pull Stock Price Every 10 Seconds
print("\n\n Pull Stock Price Every 10 Seconds .....")
symbol = "TSLA"
while True:
    try:
        # Create request for the latest trade
        request_params = StockLatestTradeRequest(symbol_or_symbols=symbol)

        # Get latest trade data (includes price)
        latest_trade = stock_data_client.get_stock_latest_trade(request_params)

        # Extract price (trade.price) and timestamp
        trade = latest_trade[symbol]
        print(f"Time: {trade.timestamp}, Price: ${trade.price}")

    except Exception as e:
        print(f"Error fetching price: {e}")

    # Wait for 10 seconds
    time.sleep(5)


# from alpaca_trade_api.common import URL
# from alpaca_trade_api.stream import Stream

# async def trade_callback(t):
#     print('trade', t)


# async def quote_callback(q):
#     print('quote', q)


# # Initiate Class Instance
# stream = Stream(TRADING_API_KEY,
#                 TRADING_API_SECRET,
#                 base_url=URL('https://paper-api.alpaca.markets'),
#                 data_feed='iex')  # <- replace to 'sip' if you have PRO subscription

# # subscribing to event
# stream.subscribe_trades(trade_callback, 'AAPL')
# stream.subscribe_quotes(quote_callback, 'IBM')

# stream.run()