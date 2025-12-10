"""
Alpaca Algo Trader (Static tickers, official alpaca-py TradingClient)
===================================================================

Strategy:
1. Use a static list of tickers (subset of S&P 500).
2. BUY: If last close < SMA20 * 0.9 (10% below moving average).
3. SELL: For all open positions, if profit > +5% → SELL ALL.

------------------------------------
Run
------------------------------------
Dry run (no trades):
   python main.py --dry-run

Live (paper orders):
   python main.py

"""
from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError

# ---------------- Config ---------------- #
SMA_LEN = 100          # SMA period
BUY_DISCOUNT = 0.95   # Buy if price < SMA * 0.98 (2% below SMA)
SELL_PROFIT = 0.1    # Sell if profit > 5%
NOTIONAL = 1000.0      # $ amount per buy 

# Static ticker list (subset of S&P 500, good enough for testing)
STATIC_TICKERS = [
 'AAPL',  # Apple Inc.
 'XOM',   # Exxon Mobil Corporation
 'JNJ',   # Johnson & Johnson
 'WMT',   # Walmart Inc.
 'MSFT',  # Microsoft Corporation
 'PG',    # Procter & Gamble Co.
 'KO',    # The Coca-Cola Company
 'INTC',  # Intel Corporation
 'CVX',   # Chevron Corporation
 'BA',    # The Boeing Company
 'UNH',   # UnitedHealth Group Incorporated
 'JPM',   # JPMorgan Chase & Co.
 'IBM',   # International Business Machines Corporation
 'GE',    # General Electric Company
 'MRK',   # Merck & Co., Inc.
 'PEP',   # PepsiCo, Inc.
 'MCD',   # McDonald’s Corporation
 'CAT',   # Caterpillar Inc.
 'COST',  # Costco Wholesale Corporation
 'DD',    # DuPont de Nemours, Inc.
 'HD',    # The Home Depot, Inc.
 'T',     # AT&T Inc.
 'AMGN',  # Amgen Inc.
 'CVS',   # CVS Health Corporation
 'LMT',   # Lockheed Martin Corporation
 'UPS',   # United Parcel Service, Inc.
 'WFC',   # Wells Fargo & Company
 'MMM',   # 3M Company
 'SLB',   # Schlumberger Limited
 'TXN',   # Texas Instruments Incorporated
 'BK',    # The Bank of New York Mellon Corporation
 'AXP',   # American Express Company
 'BMY',   # Bristol-Myers Squibb Company
 'GILD',  # Gilead Sciences, Inc.
 'GS',    # The Goldman Sachs Group, Inc.
 'AMT',   # American Tower Corporation
 'C',     # Citigroup Inc.
 'COP',   # ConocoPhillips
 'SPG',   # Simon Property Group, Inc.
 'FDX',   # FedEx Corporation
 'DD',    # DuPont de Nemours (duplicate, remove one)
 'EMR',   # Emerson Electric Co.
 'MO',    # Altria Group, Inc.
 'GE',    # General Electric Company (duplicate, remove one)
 'PFE',   # Pfizer Inc.
 'ORCL',  # Oracle Corporation
 'GPN',   # Global Payments Inc.
 'CVX',   # Chevron Corporation (duplicate, remove one)
]


def load_client() -> TradingClient:
    load_dotenv()
    key = (os.getenv("APCA_API_KEY_ID") or "").strip()
    secret = (os.getenv("APCA_API_SECRET_KEY") or "").strip()
    if not key or not secret:
        raise SystemExit("Missing API keys in .env")
    return TradingClient(key, secret, paper=True)


def get_sma_and_price(symbol: str):
    """Return (sma_value, last_close) or (None, None) if unavailable."""
    try:
        y_sym = symbol.replace('.', '-')  
        df = yf.download(
            y_sym,
            period="6mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        if df.empty or len(df) < SMA_LEN:
            return None, None
        close = df["Close"].astype(float)
        sma = close.rolling(SMA_LEN).mean()
        last_close = close.iloc[-1].item()
        last_sma = sma.iloc[-1].item()

        if np.isnan(last_sma):
            return None, None
        return last_sma, last_close
    except Exception:
        return None, None


def place_order(client: TradingClient, symbol: str, side: str, qty: int | None = None, dry=False):
    if dry:
        print(f"DRYRUN {side.upper()} {symbol} qty={qty}")
        return None
    try:
        order = client.submit_order(MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        ))
        print(f"{side.upper()} {symbol} placed: {order.id}")
        return order
    except APIError as e:
        print(f"Order error for {symbol}: {e}")
        return None


def run_strategy(client: TradingClient, dry: bool):
    tickers = STATIC_TICKERS
    print(f"Checking {len(tickers)} symbols...")

    buys = []

    for sym in tickers:
        sma, price = get_sma_and_price(sym)
        if sma is None or price is None:
            continue
        if price < (sma * BUY_DISCOUNT):
            qty = int(NOTIONAL // price)
            if qty >= 1:
                buys.append(sym)
                place_order(client, sym, 'buy', qty=qty, dry=dry)


    sells = []
    try:
        positions = client.get_all_positions()
    except APIError as e:
        print("Warning: could not fetch positions:", e)
        positions = []

    for pos in positions:
        sym = pos.symbol
        avg = float(pos.avg_entry_price)
        qty = int(float(pos.qty))
        price = float(pos.current_price)
        if avg <= 0 or qty <= 0:
            continue
        profit_pct = (price - avg) / avg
        if profit_pct > SELL_PROFIT:
            sells.append((sym, qty, profit_pct))
            place_order(client, sym, 'sell', qty=qty, dry=dry)

    # Summary
    print("\nSummary:")
    print(f"  Buy signals: {len(buys)} -> {buys[:10]}{'...' if len(buys)>10 else ''}")
    print(f"  Sell signals: {len(sells)} -> {[s[0] for s in sells][:10]}{'...' if len(sells)>10 else ''}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    client = load_client()
    acct = client.get_account()
    print("Connected to Alpaca | Account status:", acct.status)

    run_strategy(client, dry=args.dry_run)


if __name__ == '__main__':
    main()
