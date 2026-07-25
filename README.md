# AlpacaTrader — Simple Algorithmic Trading Bot

AlpacaTrader is a Python-based algorithmic trading bot that interacts with the **Alpaca API** to trade financial instruments programmatically.

It demonstrates automated trading logic built on top of Alpaca's REST and trading APIs, supporting order execution, paper trading, backtesting, and simple strategy experimentation on market data.

---

## 🚀 Project Overview

AlpacaTrader includes:

- **main.py** — Entry point for the trading bot
- **ml_trader.py** — Machine-learning-augmented trading strategy
- **backtest.py** — Backtesting harness for evaluating strategies on historical data
- **config.py** — Configuration settings (API keys, environment, symbols)
- **playground.py** — Sandbox space for strategy testing
- **test.py / test2.py** — Quick scripts for rule testing

This repository supports both simple rule-based strategies and more advanced, model-driven approaches.

> **Note:** An earlier Post-Earnings Announcement Drift (PEAD) engine was removed from this project. The empty `PEAD.py` placeholder is a leftover from that experiment and is no longer used.

---

## 🧠 Dependencies

Install the required libraries with:

```bash
pip install -r requirements.txt
```

Core dependencies include the [`alpaca-py`](https://github.com/alpacahq/alpaca-py) SDK plus the usual data-science stack (`pandas`, `numpy`, etc.).

---

## ⚙️ Configuration

Trading credentials are read from `config.py`. Provide your Alpaca API key and secret (paper-trading keys are recommended while developing):

```python
API_KEY = "your-alpaca-key"
API_SECRET = "your-alpaca-secret"
BASE_URL = "https://paper-api.alpaca.markets"
```

Never commit real API keys — keep them out of version control.

---

## ▶️ Usage

Run the bot:

```bash
python main.py
```

Backtest a strategy:

```bash
python backtest.py
```

---

## ⚠️ Disclaimer

This project is for educational and experimental purposes only. It is **not** financial advice. Automated trading carries real financial risk — use paper trading and test thoroughly before deploying any strategy with real capital.
