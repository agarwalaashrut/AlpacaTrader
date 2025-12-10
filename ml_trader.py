import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ==================== Sharpe Ratio Calculation ====================

def calculate_sharpe_ratio(returns_series, risk_free_rate=0.02, periods_per_year=252):
    """
    Calculate Sharpe ratio for a returns series.
    
    Parameters:
    - returns_series: pd.Series of daily returns
    - risk_free_rate: annualized risk-free rate (default 2%)
    - periods_per_year: trading days per year (default 252)
    
    Returns:
    - sharpe_ratio: annualized Sharpe ratio
    """
    excess_returns = returns_series - (risk_free_rate / periods_per_year)
    if excess_returns.std() == 0:
        return 0.0
    sharpe = np.sqrt(periods_per_year) * (excess_returns.mean() / excess_returns.std())
    return sharpe

# ==================== Data & Features ====================

assets = [
    "SPY", "QQQ", "IWM",    # equities
    "TLT", "IEF",           # bonds
    "GLD", "DBC",           # commodities
    "UUP"                   # USD index
]

start = "2005-01-01"
end   = "2025-01-01"
data = yf.download(assets, start=start, end=end)
prices = data.get("Adj Close")

if prices is None:
    print("Adj Close not found. Falling back to Close prices.")
    prices = data["Close"]

returns = prices.pct_change().dropna()
def make_features(prices, returns):
    df = pd.DataFrame(index=returns.index)

    # Momentum windows
    for w in [21, 63, 126]:
        df[f"mom_{w}"] = returns.rolling(w).mean()
    
    # Volatility windows
    for w in [21, 63]:
        df[f"vol_{w}"] = returns.rolling(w).std()
    
    # Skew & Kurtosis
    df["skew_63"] = returns.rolling(63).skew()
    df["kurt_63"] = returns.rolling(63).kurt()

    # RSI example (simplified)
    delta = prices.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    df["rsi_14"] = 100 - 100 / (1 + up / down)

    return df.dropna()
features = {asset: make_features(prices[asset], returns[asset]) for asset in assets}
horizon = 5

predictions = {}

for asset in assets:
    df = features[asset].copy()
    df["future_ret"] = returns[asset].shift(-horizon)

    df = df.dropna()

    X = df.drop("future_ret", axis=1).values
    y = df["future_ret"].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Time-series split for finance
    tscv = TimeSeriesSplit(n_splits=5)

    model = ElasticNetCV(cv=tscv, l1_ratio=[0.1, 0.5, 0.9], n_jobs=-1)
    model.fit(X_scaled, y)

    df["prediction"] = model.predict(X_scaled)

    predictions[asset] = df[["prediction", "future_ret"]]
pred_df = pd.concat(
    {asset: predictions[asset]["prediction"] for asset in assets},
    axis=1
).dropna()
spy_vol = returns["SPY"].rolling(21).std()
regime = (spy_vol < spy_vol.rolling(252).mean()).astype(int)

# Align with predictions
regime = regime.reindex(pred_df.index).fillna(0)
volatility = returns.rolling(21).std()
vol_small = volatility.reindex(pred_df.index)

weights = pd.DataFrame(0, index=pred_df.index, columns=assets)

for date in pred_df.index:
    preds = pred_df.loc[date]
    vols = vol_small.loc[date]

    # Risk-adjusted forecast
    score = preds / vols
    score = score.replace([np.inf, -np.inf], np.nan).dropna()

    # Select top 3 assets
    top_assets = score.nlargest(3).index

    if regime.loc[date] == 0:
        continue  # skip if volatility regime is bad

    w = 1 / len(top_assets)
    weights.loc[date, top_assets] = w
aligned_returns = returns.reindex(weights.index)

strategy_ret = (weights.shift(1) * aligned_returns).sum(axis=1)
strategy_cum = (1 + strategy_ret).cumprod()

buy_hold_cum = (1 + returns["SPY"]).cumprod()

# ==================== Performance Metrics ====================

strategy_sharpe = calculate_sharpe_ratio(strategy_ret)
spy_sharpe = calculate_sharpe_ratio(returns["SPY"])

strategy_total_return = (strategy_cum.iloc[-1] - 1) * 100
spy_total_return = (buy_hold_cum.iloc[-1] - 1) * 100

strategy_max_dd = (strategy_cum / strategy_cum.cummax() - 1).min() * 100
spy_max_dd = (buy_hold_cum / buy_hold_cum.cummax() - 1).min() * 100

print(f"\n{'='*60}")
print(f"{'STRATEGY PERFORMANCE METRICS':^60}")
print(f"{'='*60}")
print(f"{'Metric':<30} {'ML Strategy':>15} {'SPY B&H':>15}")
print(f"{'-'*60}")
print(f"{'Sharpe Ratio':<30} {strategy_sharpe:>15.2f} {spy_sharpe:>15.2f}")
print(f"{'Total Return':<30} {strategy_total_return:>14.1f}% {spy_total_return:>14.1f}%")
print(f"{'Max Drawdown':<30} {strategy_max_dd:>14.1f}% {spy_max_dd:>14.1f}%")
print(f"{'='*60}\n")

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(strategy_cum, label="ML Momentum Strategy")
plt.plot(buy_hold_cum, label="SPY Buy & Hold")
plt.legend()
plt.title("Machine Learning Momentum Strategy vs SPY")
plt.grid(True)
plt.show()
