"""
sensitivity.py
--------------
Parameter sensitivity analysis. Sweeps a single parameter across a range
of values, runs a full backtest for each, and collects key metrics to
show whether a strategy's performance is robust or fragile to that parameter.
"""

import pandas as pd
import numpy as np

from .backtest import Backtest
from . import performance

def run_sensitivity_sweep(base_config, param_name, param_values, benchmark_equity=None, risk_free=0.0, progress_callback=None) -> pd.DataFrame:
    results = []
    n_total = len(param_values)
    
    for i, value in enumerate(param_values):
        cfg = base_config.copy()
        builder = cfg.pop('strategy_factory_builder')
        cfg['strategy_factory'] = builder(value)
        
        bt = Backtest(**cfg)
        bt_results = bt.run()
        if len(bt_results['equity']) < 2:
            raise ValueError(f"Backtest for {param_name}={value} produced no equity curve.")
        
        equity = bt_results['equity']
        summary = performance.summarize(equity, benchmark_equity=benchmark_equity, risk_free=risk_free)
        
        summary['# Trades'] = bt_results['n_trades']
        summary['Final Equity'] = bt_results['final_equity']
        
        results.append(summary)
        
        if progress_callback:
            progress_callback(i, n_total)
            
    df = pd.DataFrame(results, index=pd.Index(param_values))
    df.index.name = param_name
    
    columns_to_keep = ['Sharpe', 'CAGR', 'Max Drawdown', 'Annualized Vol', '# Trades', 'Final Equity']
    return df[columns_to_keep]

def interpret_sensitivity(results_df, param_name) -> str:
    """
    Plain-English verdict on a sweep, judged on Sharpe ratio:
      * stable : the Sharpe range is small relative to its level
      * trend  : Sharpe rises or falls consistently with the parameter
                 (Spearman rank correlation, so one small wobble doesn't hide it)
      * peak   : otherwise; reports the best value and how wide the good zone is
    Always notes whether Sharpe stays positive across the range.
    """
    sharpe = results_df['Sharpe'].dropna()
    if len(sharpe) < 2:
        return "Not enough values to judge robustness; widen the range."
    lo, hi = sharpe.min(), sharpe.max()
    if lo > 0:
        sign_note = "Sharpe stays positive across the whole range."
    elif hi < 0:
        sign_note = "Sharpe is negative across the whole range."
    else:
        sign_note = "Sharpe turns negative for some values."

    if hi - lo < max(0.10, 0.20 * abs(sharpe.mean())):
        return (f"Stable: Sharpe stays between {lo:.2f} and {hi:.2f} for every {param_name} tested. "
                f"{sign_note}")

    rho = pd.Series(sharpe.index, index=sharpe.index, dtype=float).corr(sharpe, method="spearman")
    first, last = sharpe.iloc[0], sharpe.iloc[-1]
    if rho >= 0.8:
        return f"Improves as {param_name} increases (Sharpe {first:.2f} up to {last:.2f}). {sign_note}"
    if rho <= -0.8:
        return f"Gets worse as {param_name} increases (Sharpe {first:.2f} down to {last:.2f}). {sign_note}"

    best_value, best_sharpe = sharpe.idxmax(), hi
    good = sharpe[sharpe >= best_sharpe - 0.25 * (hi - lo)].index
    if len(good) == 1:
        return (f"Sharp peak at {param_name} = {best_value} (Sharpe {best_sharpe:.2f}); results fall off "
                f"on both sides, a sign the setting may be over-tuned. {sign_note}")
    return (f"Best around {param_name} = {best_value} (Sharpe {best_sharpe:.2f}), with similar results "
            f"from {min(good)} to {max(good)}. {sign_note}")
