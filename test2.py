from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
import os, requests
from pathlib import Path
from dotenv import load_dotenv


load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
KEY = (os.getenv("APCA_API_KEY_ID") or "").strip()
SEC = (os.getenv("APCA_API_SECRET_KEY") or "").strip()

trading_client = TradingClient(KEY, SEC, paper=True)

# Get our account information.
account = trading_client.get_account()

# Check if our account is restricted from trading.
if account.trading_blocked:
    print('Account is currently restricted from trading.')

# Check how much money we can use to open new positions.
print('${} is available as buying power.'.format(account.buying_power))