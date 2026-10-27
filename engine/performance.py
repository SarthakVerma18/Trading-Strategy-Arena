"""
performance.py
---------------
Post-hoc performance analytics computed from the Portfolio's equity curve.

Metrics:
  - Total / annualized return, CAGR
  - Annualized volatility
  - Sharpe ratio (excess return / vol, annualized)
  - Sortino ratio (downside-deviation adjusted)
  - Max drawdown, Calmar ratio
  - Alpha / Beta vs. a benchmark (OLS of strategy excess returns on
    benchmark excess returns -- the standard CAPM-style regression)
  - Rolling historical & parametric Value-at-Risk (VaR)
"""

from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def returns_from_equity(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = max(len(equity) / TRADING_DAYS, 1e-9)
    return (1 + total_return) ** (1 / years) - 1


def annualized_vol(rets: pd.Series) -> float:
    return rets.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(rets) > 1 else 0.0


def sharpe_ratio(rets: pd.Series, risk_free: float = 0.0) -> float:
    if len(rets) < 2 or rets.std(ddof=1) == 0:
        return 0.0
    excess = rets - risk_free / TRADING_DAYS
    return np.sqrt(TRADING_DAYS) * excess.mean() / rets.std(ddof=1)


def sortino_ratio(rets: pd.Series, risk_free: float = 0.0) -> float:
    if len(rets) < 2:
        return 0.0
    excess = rets - risk_free / TRADING_DAYS
    downside = rets[rets < 0]
    dd = downside.std(ddof=1)
    if not dd or np.isnan(dd) or dd == 0:
        return 0.0
    return np.sqrt(TRADING_DAYS) * excess.mean() / dd


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return dd.min()


def drawdown_series(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return equity / running_max - 1.0


def calmar_ratio(equity: pd.Series) -> float:
    mdd = abs(max_drawdown(equity))
    if mdd < 1e-9:
        return 0.0
    return cagr(equity) / mdd


def alpha_beta(strategy_rets: pd.Series, benchmark_rets: pd.Series, risk_free: float = 0.0):
    """
    OLS regression: (r_strategy - rf) = alpha + beta * (r_benchmark - rf) + eps
    Returns (alpha_annualized, beta).
    """
    aligned = pd.concat([strategy_rets, benchmark_rets], axis=1, join="inner").dropna()
    aligned.columns = ["strat", "bench"]
    if len(aligned) < 5 or aligned["bench"].var() == 0:
        return 0.0, 0.0
    rf_daily = risk_free / TRADING_DAYS
    y = aligned["strat"] - rf_daily
    x = aligned["bench"] - rf_daily
    beta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    alpha_daily = y.mean() - beta * x.mean()
    alpha_annual = alpha_daily * TRADING_DAYS
    return alpha_annual, beta


def rolling_var(rets: pd.Series, window: int = 20, confidence: float = 0.95, method: str = "historical") -> pd.Series:
    """
    Rolling Value-at-Risk, expressed as a POSITIVE number = fraction of
    equity at risk over 1 day at the given confidence level.

    method='historical' : empirical quantile of trailing returns.
    method='parametric'  : Gaussian VaR using trailing mean/std (z-score).
    """
    if method == "parametric":
        from scipy.stats import norm
        z = norm.ppf(1 - confidence)
        roll_mean = rets.rolling(window).mean()
        roll_std = rets.rolling(window).std()
        var = -(roll_mean + z * roll_std)
    else:
        var = rets.rolling(window).apply(lambda x: -np.percentile(x, (1 - confidence) * 100), raw=True)
    return var.clip(lower=0) + 0.0  # + 0.0 normalizes -0.0 to 0.0


def summarize(equity: pd.Series, benchmark_equity: pd.Series = None, risk_free: float = 0.0,
              var_window: int = 20, var_confidence: float = 0.95) -> dict:
    rets = returns_from_equity(equity)
    out = {
        "Total Return": equity.iloc[-1] / equity.iloc[0] - 1 if len(equity) > 1 else 0.0,
        "CAGR": cagr(equity),
        "Annualized Vol": annualized_vol(rets),
        "Sharpe": sharpe_ratio(rets, risk_free),
        "Sortino": sortino_ratio(rets, risk_free),
        "Max Drawdown": max_drawdown(equity),
        "Calmar": calmar_ratio(equity),
    }
    if benchmark_equity is not None and len(benchmark_equity) > 1:
        bench_rets = returns_from_equity(benchmark_equity)
        alpha, beta = alpha_beta(rets, bench_rets, risk_free)
        out["Alpha (ann.)"] = alpha
        out["Beta"] = beta
    else:
        out["Alpha (ann.)"] = np.nan
        out["Beta"] = np.nan

    var_series = rolling_var(rets, window=var_window, confidence=var_confidence)
    out[f"VaR {int(var_confidence*100)}% (latest, 1d)"] = var_series.dropna().iloc[-1] if var_series.dropna().shape[0] else np.nan
    out[f"VaR {int(var_confidence*100)}% (avg, 1d)"] = var_series.dropna().mean() if var_series.dropna().shape[0] else np.nan
    return out


# ---------- Strategy diversification analytics ----------

def strategy_correlation_matrix(equity_dict: dict) -> pd.DataFrame:
    """
    Correlation matrix of daily returns across strategies.

    equity_dict: {label: pd.Series(equity)} — one entry per strategy.
    Returns a DataFrame correlation matrix (labels × labels).
    """
    rets = {}
    for label, eq in equity_dict.items():
        if len(eq) > 2:
            rets[label] = returns_from_equity(eq)
    if len(rets) < 2:
        return pd.DataFrame()
    df = pd.DataFrame(rets)
    return df.corr()


def combined_portfolio_equity(equity_dict: dict, weights: dict,
                               initial_capital: float = 1_000_000.0) -> pd.Series:
    """
    Synthesize an equity curve from a weighted combination of strategy returns.

    equity_dict : {label: pd.Series(equity)}
    weights     : {label: float}  — fractions summing to ~1.0
    Returns a pd.Series equity curve for the combined portfolio.
    """
    rets_list = []
    w_list = []
    for label, eq in equity_dict.items():
        if label in weights and len(eq) > 2:
            r = returns_from_equity(eq)
            rets_list.append(r)
            w_list.append(weights[label])
    if not rets_list:
        return pd.Series(dtype=float)
    combined = pd.concat(rets_list, axis=1, join="inner")
    combined.columns = range(len(combined.columns))
    w_arr = np.array(w_list)
    w_arr = w_arr / w_arr.sum()  # normalize to exactly 1.0
    port_rets = combined.values @ w_arr
    equity = initial_capital * np.cumprod(1 + port_rets)
    return pd.Series(equity, index=combined.index, name="combined_equity")


def diversification_ratio(equity_dict: dict, weights: dict) -> float:
    """
    Diversification ratio = (weighted average of individual volatilities) /
                            (volatility of the combined portfolio).
    A ratio > 1.0 means diversification is helping; 1.0 means no benefit.
    """
    rets = {}
    w_list = []
    for label, eq in equity_dict.items():
        if label in weights and len(eq) > 2:
            rets[label] = returns_from_equity(eq)
            w_list.append(weights[label])
    if len(rets) < 2:
        return 1.0
    w_arr = np.array(w_list)
    w_arr = w_arr / w_arr.sum()
    vols = np.array([r.std() for r in rets.values()])
    weighted_avg_vol = np.dot(w_arr, vols)
    combined = pd.concat(list(rets.values()), axis=1, join="inner")
    port_rets = combined.values @ w_arr
    port_vol = np.std(port_rets, ddof=1)
    if port_vol < 1e-12:
        return 1.0
    return weighted_avg_vol / port_vol
