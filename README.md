# Trading-Strategy-Arena
Trading Strategy Arena is a Python-based platform for simulating and evaluating trading strategies on historical market data. It provides a simple user interface and a modular backend to run strategy tournaments, backtests, and performance analysis.<br/>

Features:<br/>
  Simulated backtesting with support for daily trade execution and slippage.<br/>
  Built-in metrics like Sharpe ratio, alpha, and maximum drawdown.<br/>
  Supports a variety of strategies including momentum, mean-reversion, and custom ML-based agents.<br/>
  Tkinter-based GUI for launching backtests and viewing results.<br/>
  Modular design to easily add new strategies or modify existing ones.<br/>

## Project Structure
trading-strategy-arena/<br/>
│
├── main_gui.py          # Launches the user interface<br/>
├── engine/              # Core backtesting logic<br/>
├── strategies/          # Template and sample strategies<br/>
├── utils/               # Helper modules (plotting, metrics, etc.)<br/>
