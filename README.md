# Trading-Strategy-Arena
Trading Strategy Arena is a Python-based platform for simulating and evaluating trading strategies on historical market data. It provides a simple user interface and a modular backend to run strategy tournaments, backtests, and performance analysis.

Features:
  Simulated backtesting with support for daily trade execution and slippage.\ 
  Built-in metrics like Sharpe ratio, alpha, and maximum drawdown.\ 
  Supports a variety of strategies including momentum, mean-reversion, and custom ML-based agents.
  Tkinter-based GUI for launching backtests and viewing results.
  Modular design to easily add new strategies or modify existing ones.

## Project Structure
trading-strategy-arena/
│
├── main_gui.py          # Launches the user interface
├── engine/              # Core backtesting logic
├── strategies/          # Template and sample strategies
├── utils/               # Helper modules (plotting, metrics, etc.)
