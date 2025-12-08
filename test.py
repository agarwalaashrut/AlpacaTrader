# preflight_alpaca.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env that sits next to this script (robust on Windows/OneDrive)
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.common.exceptions import APIError

KEY = os.getenv("APCA_API_KEY_ID")
SEC = os.getenv("APCA_API_SECRET_KEY")

def mask(s): 
    return s if not s else s[:4] + "*"*(len(s)-8) + s[-4:]

print("Loaded KEY:", mask(KEY))
print("Loaded SEC:", mask(SEC))

try:
    # paper=True guarantees correct paper endpoint
    client = TradingClient(KEY, SEC, paper=True)
    acct = client.get_account()
    print("Account status:", acct.status)      # 'ACTIVE' expected for paper
    print("Trading blocked?:", acct.trading_blocked)
    # tiny data call to verify auth fully works
    _ = client.get_all_positions()             # should return list (possibly empty)
    print("Preflight OK ✅")
except APIError as e:
    print("APIError ❌:", e)                   # prints precise reason from Alpaca
except Exception as e:
    print("Other error ❌:", repr(e))
