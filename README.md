# AlpacaTrader — Simple Algorithmic Trading Bot

AlpacaTrader is a Python-based algorithmic trading bot that interacts with the **Alpaca API** to trade financial instruments programmatically.

This project demonstrates automated trading logic built on top of Alpaca’s REST and trading APIs, allowing order execution, paper trading, backtesting, and simple strategy experimentation using market data. Alpaca’s API is a developer-first trading platform that supports programmatic orders, account management, and market data. :contentReference[oaicite:1]{index=1}

---

## 🚀 Project Overview

AlpacaTrader includes:

- **main.py** — Entry point for the trading bot  
- **PEAD.py** — Implementation of Post-Earnings Announcement Drift logic  
- **ml_trader.py** — Machine-learning-augmented trading strategy  
- **backtest.py** — Backtesting example on historical data  
- **config.py** — Configuration settings (API keys, environment, symbols)  
- **playground.py** — Sandbox space for strategy testing  
- **test.py / test2.py** — Quick scripts for rule testing  

This repository is structured to support both simple rule-based strategies and more advanced, model-driven approaches.

---

## 🧠 Dependencies

Install the required libraries with:

```bash
pip install -r requirements.txt
