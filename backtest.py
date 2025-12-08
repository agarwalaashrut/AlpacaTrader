"""
Production-Grade Backtesting Engine for Alpaca Trading Strategy
================================================================

Features:
- No lookahead bias (next-day execution)
- Unified trading calendar across all symbols
- FIFO position accounting
- Transaction costs and slippage modeling
- Consistent capital allocation
- Proper rolling SMA window handling
- Realistic position valuation with forward-fill
- Comprehensive performance metrics
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum

# Import strategy parameters from main
from main import STATIC_TICKERS, SMA_LEN, BUY_DISCOUNT, SELL_PROFIT, NOTIONAL


class OrderType(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Trade:
    """Represents a single trade execution."""
    date: pd.Timestamp
    symbol: str
    order_type: OrderType
    quantity: int
    price: float
    commission: float
    slippage: float
    total_cost: float
    
    def __repr__(self):
        return f"{self.date.date()} | {self.order_type.value:4s} | {self.symbol:6s} | {self.quantity:6d} @ ${self.price:8.2f} | Cost: ${self.total_cost:10.2f}"


@dataclass
class Position:
    """FIFO position tracking."""
    symbol: str
    fifo_queue: List[Tuple[int, float]]  # (quantity, avg_price)
    
    @property
    def total_quantity(self) -> int:
        return sum(qty for qty, _ in self.fifo_queue)
    
    @property
    def avg_cost(self) -> float:
        if not self.fifo_queue:
            return 0.0
        total_qty = sum(qty for qty, _ in self.fifo_queue)
        total_cost = sum(qty * price for qty, price in self.fifo_queue)
        return total_cost / total_qty if total_qty > 0 else 0.0
    
    def add(self, quantity: int, price: float):
        """Add shares to position (FIFO)."""
        self.fifo_queue.append((quantity, price))
    
    def remove(self, quantity: int, execution_price: float) -> float:
        """Remove shares from position (FIFO) and return realized PnL."""
        remaining = quantity
        realized_pnl = 0.0
        
        while remaining > 0 and self.fifo_queue:
            qty, cost_basis = self.fifo_queue[0]
            if qty <= remaining:
                remaining -= qty
                # Realized PnL = (execution_price - cost_basis) * qty
                realized_pnl += (execution_price - cost_basis) * qty
                self.fifo_queue.pop(0)
            else:
                # Partial fill
                realized_pnl += (execution_price - cost_basis) * remaining
                self.fifo_queue[0] = (qty - remaining, cost_basis)
                remaining = 0
        
        return realized_pnl
    
    def is_empty(self) -> bool:
        return self.total_quantity == 0


class BacktestEngine:
    """Production-grade backtesting engine."""
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,  # 0.1% commission rate per trade
        slippage_pct: float = 0.0005,  # 0.05% slippage
        risk_free_rate: float = 0.02
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate  # Rate, not per-trade flat fee
        self.slippage_pct = slippage_pct
        self.risk_free_rate = risk_free_rate
        
        # State tracking
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.daily_returns: List[float] = []
        
    def get_unified_calendar(self, symbols: List[str], years: int) -> Tuple[pd.DatetimeIndex, Dict[str, pd.DataFrame]]:
        """
        Get unified trading calendar (union of available dates for all symbols).
        This prevents data shrinkage from misaligned calendars.
        """
        print(f"Fetching {years}-year historical data for {len(symbols)} symbols...")
        end_date = datetime(year=2024, month=12, day=31)
        start_date = end_date - timedelta(days=years * 365)
        
        # Download data for all symbols
        data = {}
        valid_symbols = []
        
        for symbol in symbols:
            try:
                df = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    interval="1d",
                    auto_adjust=True,
                    progress=False
                )
                if not df.empty and len(df) > SMA_LEN:
                    data[symbol] = df
                    valid_symbols.append(symbol)
                else:
                    print(f"⚠ Skipped {symbol}: insufficient data")
            except Exception as e:
                print(f"⚠ Failed to fetch {symbol}: {e}")
        
        if not valid_symbols:
            raise ValueError("No valid symbols with sufficient data")
        
        # Use UNION of all available dates (not intersection) to prevent data loss
        # Forward-fill missing prices to handle gaps
        all_dates = pd.DatetimeIndex([])
        for df in data.values():
            all_dates = all_dates.union(df.index)
        
        all_dates = all_dates.sort_values()
        
        print(f"✓ Loaded data for {len(valid_symbols)} symbols")
        print(f"✓ Unified calendar: {len(all_dates)} trading days ({all_dates[0].date()} to {all_dates[-1].date()})")
        
        return all_dates, {symbol: data[symbol] for symbol in valid_symbols}
    
    def get_price(self, df: pd.DataFrame, date: pd.Timestamp) -> Optional[float]:
        """
        Get price for a date, with backward-fill for missing data.
        Only uses prices on or before the given date (no lookahead).
        """
        if date in df.index:
            return float(df.loc[date, 'Close'])
        
        # Only backward-fill: find the last available price on or before this date
        valid_prices = df[df.index <= date]
        if not valid_prices.empty:
            return float(valid_prices['Close'].iloc[-1])
        
        return None
    
    def calculate_sma(self, df: pd.DataFrame, date: pd.Timestamp, window: int) -> Optional[float]:
        """
        Calculate SMA up to and including the given date (no lookahead).
        """
        cutoff_df = df[df.index <= date]
        if len(cutoff_df) < window:
            return None
        return float(cutoff_df['Close'].tail(window).mean())
    
    def generate_signals(
        self,
        date: pd.Timestamp,
        symbol_data: Dict[str, pd.DataFrame]
    ) -> List[Tuple[str, OrderType, int]]:
        """
        Generate buy/sell signals for all symbols on a given date.
        Returns list of (symbol, order_type, max_quantity)
        """
        signals = []
        
        for symbol, df in symbol_data.items():
            price = self.get_price(df, date)
            if price is None:
                continue
            
            sma = self.calculate_sma(df, date, SMA_LEN)
            if sma is None:
                continue
            
            # BUY SIGNAL: Price dips below SMA * BUY_DISCOUNT
            if price < (sma * BUY_DISCOUNT) and symbol not in self.positions:
                # Calculate maximum quantity based on NOTIONAL
                max_qty = int(NOTIONAL / price)
                if max_qty > 0:
                    signals.append((symbol, OrderType.BUY, max_qty))
            
            # SELL SIGNAL: Profit target reached
            if symbol in self.positions:
                position = self.positions[symbol]
                unrealized_pnl_pct = (price - position.avg_cost) / position.avg_cost
                
                if unrealized_pnl_pct >= SELL_PROFIT:
                    signals.append((symbol, OrderType.SELL, position.total_quantity))
        
        return signals
    
    def execute_trade(
        self,
        date: pd.Timestamp,
        symbol: str,
        order_type: OrderType,
        quantity: int,
        price: float
    ) -> bool:
        """
        Execute a trade with commission and slippage modeling.
        """
        # Apply slippage
        slippage = price * self.slippage_pct
        execution_price = price + slippage if order_type == OrderType.BUY else price - slippage
        
        # Calculate total cost
        trade_value = execution_price * quantity
        commission = trade_value * self.commission_rate
        total_cost = trade_value + commission
        
        if order_type == OrderType.BUY:
            # Check available capital
            if total_cost > self.cash:
                return False
            
            # Execute buy
            self.cash -= total_cost
            if symbol not in self.positions:
                self.positions[symbol] = Position(symbol, [])
            self.positions[symbol].add(quantity, execution_price)
            
        else:  # SELL
            # Check position exists and has enough shares
            if symbol not in self.positions or self.positions[symbol].total_quantity < quantity:
                return False
            
            # Execute sell and compute realized PnL
            realized_pnl = self.positions[symbol].remove(quantity, execution_price)
            self.cash += trade_value - commission
            
            if self.positions[symbol].is_empty():
                del self.positions[symbol]
        
        # Record trade
        trade = Trade(
            date=date,
            symbol=symbol,
            order_type=order_type,
            quantity=quantity,
            price=execution_price,
            commission=commission,
            slippage=slippage * quantity,
            total_cost=total_cost if order_type == OrderType.BUY else trade_value - commission
        )
        self.trades.append(trade)
        
        return True
    
    def calculate_portfolio_value(
        self,
        date: pd.Timestamp,
        symbol_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Calculate total portfolio value (cash + positions)."""
        value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol not in symbol_data:
                continue
            
            price = self.get_price(symbol_data[symbol], date)
            if price is None:
                continue
            
            value += price * position.total_quantity
        
        return value
    
    def run_backtest(self, years: int = 10) -> Dict:
        """
        Run the complete backtest with next-day execution.
        
        Process:
        1. On date t: Generate signals using data up to t
        2. On date t+1: Execute pending signals at t+1's price
        3. Record portfolio value and returns
        """
        # Get unified calendar and data
        dates, symbol_data = self.get_unified_calendar(list(set(STATIC_TICKERS)), years)
        
        print(f"\nRunning backtest from {dates[0].date()} to {dates[-1].date()}...")
        
        # Reset state
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        
        # Buffer for pending signals (generated on day t, executed on day t+1)
        pending_signals = []
        last_portfolio_value = self.initial_capital
        
        # Process each date
        for i, date in enumerate(dates):
            if (i + 1) % 500 == 0:
                print(f"  Processing {i + 1}/{len(dates)} days...")
            
            # Execute pending signals from previous day at today's price
            for symbol, order_type, quantity in pending_signals:
                if symbol in symbol_data:
                    price = self.get_price(symbol_data[symbol], date)
                    if price is not None:
                        self.execute_trade(date, symbol, order_type, quantity, price)
            
            # Generate signals for tomorrow (only if not the last date)
            if i < len(dates) - 1:
                pending_signals = self.generate_signals(date, symbol_data)
            else:
                pending_signals = []  # No execution possible after last date
            
            # Calculate portfolio value
            portfolio_value = self.calculate_portfolio_value(date, symbol_data)
            
            # Track returns
            if portfolio_value > 0 and last_portfolio_value > 0:
                daily_return = (portfolio_value - last_portfolio_value) / last_portfolio_value
                self.daily_returns.append(daily_return)
            
            self.equity_curve.append((date, portfolio_value))
            last_portfolio_value = portfolio_value
        
        # Calculate performance metrics
        results = self._calculate_metrics()
        
        return results
    
    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if not self.equity_curve:
            return {}
        
        final_value = self.equity_curve[-1][1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Sharpe Ratio
        if len(self.daily_returns) < 2:
            sharpe_ratio = 0.0
        else:
            returns_arr = np.array(self.daily_returns)
            excess_returns = returns_arr - (self.risk_free_rate / 252)
            sharpe_ratio = np.sqrt(252) * (np.mean(excess_returns) / np.std(excess_returns, ddof=1))
        
        # Maximum Drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        # Win rate and profit factor
        buy_trades = [t for t in self.trades if t.order_type == OrderType.BUY]
        sell_trades = [t for t in self.trades if t.order_type == OrderType.SELL]
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.trades),
            'num_buy_trades': len(buy_trades),
            'num_sell_trades': len(sell_trades),
            'final_value': final_value,
            'total_commission': sum(t.commission for t in self.trades),
            'total_slippage': sum(t.slippage for t in self.trades)
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown (positive magnitude) from equity curve."""
        if not self.equity_curve:
            return 0.0
        
        values = np.array([v for _, v in self.equity_curve])
        cummax = np.maximum.accumulate(values)
        drawdown = (cummax - values) / cummax
        
        return float(np.max(drawdown))
    
    def print_results(self, results: Dict):
        """Print backtest results."""
        print("\n" + "=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)
        print(f"Initial Capital:        ${self.initial_capital:>15,.2f}")
        print(f"Final Portfolio Value:  ${results['final_value']:>15,.2f}")
        print(f"Total Return:           {results['total_return']:>15.2%}")
        print(f"\nRisk-Adjusted Returns:")
        print(f"  Sharpe Ratio:         {results['sharpe_ratio']:>15.2f}")
        print(f"  Max Drawdown:         {results['max_drawdown']:>15.2%}")
        print(f"\nTrading Activity:")
        print(f"  Total Trades:         {results['num_trades']:>15d}")
        print(f"  Buy Trades:           {results['num_buy_trades']:>15d}")
        print(f"  Sell Trades:          {results['num_sell_trades']:>15d}")
        print(f"\nCosts:")
        print(f"  Total Commission:     ${results['total_commission']:>15,.2f}")
        print(f"  Total Slippage:       ${results['total_slippage']:>15,.2f}")
        print("=" * 70)
    
    def print_trades(self, limit: int = 20):
        """Print trade history."""
        print(f"\n{'Trade History (Last ' + str(limit) + ')':^70}")
        print("-" * 130)
        for trade in self.trades[-limit:]:
            print(trade)
        print("-" * 130)
    
    def plot_results(self):
        """Plot equity curve and key metrics."""
        if not self.equity_curve:
            print("No data to plot")
            return
        
        dates = [d for d, _ in self.equity_curve]
        values = [v for _, v in self.equity_curve]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Equity curve
        ax1.plot(dates, values, linewidth=1.5, label='Portfolio Value')
        ax1.fill_between(dates, self.initial_capital, values, alpha=0.3)
        ax1.axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.set_title('Equity Curve', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Daily returns distribution
        ax2.hist(self.daily_returns, bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(x=np.mean(self.daily_returns), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.daily_returns):.4f}')
        ax2.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Daily Return')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def main():
    """Run backtest."""
    engine = BacktestEngine(
        initial_capital=100000.0,
        commission_rate=0.001,
        slippage_pct=0.0005
    )
    
    results = engine.run_backtest(years=10)
    engine.print_results(results)
    engine.print_trades(limit=30)
    engine.plot_results()


if __name__ == '__main__':
    main()