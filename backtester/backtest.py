# backtester/backtest.py

import pandas as pd
import numpy as np
from backtester.metrics import calculate_metrics
import matplotlib.pyplot as plt

def run_backtest(df: pd.DataFrame, agent) -> dict:
    """
    Runs the agent on the provided data and computes performance metrics.
    
    Args:
        df: DataFrame with at least 'Close' column.
        agent: Object with a .run(df) method that returns np.array of signals (1/-1/0)
    
    Returns:
        dict with metrics and equity curve.
    """
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column.")

    df = df.copy()
    df["Signal"] = agent.run(df)

    # NaNs to 0
    df["Signal"] = df["Signal"].fillna(0)

    # Daily returns
    df["Market_Returns"] = df["Close"].pct_change().fillna(0)
    df["Strategy_Returns"] = df["Signal"].shift(1) * df["Market_Returns"]
    df["Strategy_Returns"] = df["Strategy_Returns"].fillna(0)

    # Equity curve
    df["Equity_Curve"] = (1 + df["Strategy_Returns"]).cumprod()

    # Compute metrics
    metrics = calculate_metrics(df["Strategy_Returns"], df["Market_Returns"])

    return {
        "Sharpe Ratio": metrics["sharpe"],
        "Max Drawdown": metrics["max_drawdown"],
        "Alpha": metrics["alpha"],
        "Beta": metrics["beta"],
        "equity_curve": df["Equity_Curve"]
    }

def plot_equity(equity_series: pd.Series, title="Equity Curve"):
    """
    Plots the equity curve using matplotlib.
    """
    plt.figure(figsize=(10, 4))
    equity_series.plot()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
