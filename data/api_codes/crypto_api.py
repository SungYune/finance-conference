from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
import data.config as config
client = Client(config.binance_public, config.binance_secret)

# fetch 1 minute klines for the last day up until now
klines = client.get_historical_klines("BTCUSDP", Client.KLINE_INTERVAL_5MINUTE, "1 Dec, 2021", "1 Jan, 2022")
#get tickers
