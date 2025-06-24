# backtester/metrics.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def calculate_metrics(strategy_returns, market_returns):
    sharpe = np.mean(strategy_returns) / np.std(strategy_returns + 1e-8) * np.sqrt(252)

    # Max drawdown
    cumulative = (1 + strategy_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()

    # Alpha and Beta via linear regression
    X = market_returns.values.reshape(-1, 1)
    y = strategy_returns.values
    reg = LinearRegression().fit(X, y)
    alpha = reg.intercept_ * 252
    beta = reg.coef_[0]

    return {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "alpha": alpha,
        "beta": beta,
    }
