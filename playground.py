from alpaca.trading.client import TradingClient
import config


client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=True)
account = dict(client.get_account())
print(account)