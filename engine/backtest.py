"""
backtest.py
------------
The event loop that ties everything together. This is the "engine" in
"backtesting engine": on every iteration it drains the queue and routes
each event to whichever component handles that type, exactly mirroring
how a live trading system would be wired (which is the point of the
event-driven design -- swapping SimulatedExecutionHandler for a real
broker API later requires no change to Strategy or Portfolio code).

    while data remains:
        data_handler.update_bars()        # -> pushes 1 MarketEvent for bar t
        drain queue:
            MARKET -> execution.on_market_event(event)   # fill due orders at bar t's OPEN
                      strategy.calculate_signals(event)  # then signals from bar t's CLOSE
            FILL   -> portfolio.on_fill(event)
            SIGNAL -> portfolio.on_signal(event)          # -> may push OrderEvent
            ORDER  -> execution.on_order(event)           # queued, filled after latency
        portfolio.update_timeindex()      # mark to market at bar t's close, after
                                          # every fill of the bar (may trigger the
                                          # drawdown circuit breaker)
        drain queue                       # route any risk-liquidation orders

Fills are routed before the bar's signals, matching real time order
(open before close), so a signal is always sized against a position that
already reflects every fill that has happened.
"""

from __future__ import annotations
import os
import queue
import time
import pandas as pd

from .event import MarketEvent, SignalEvent, OrderEvent, FillEvent
from .portfolio import Portfolio
from .execution import SimulatedExecutionHandler, FrictionConfig
from . import performance


class Backtest:
    def __init__(self, data_handler_factory, strategy_factory, symbols,
                 initial_capital=1_000_000.0, friction: FrictionConfig = None,
                 risk_per_trade=0.01, label="Strategy",
                 max_drawdown_pct: float = None, max_position_pct: float = None,
                 max_gross_leverage: float = 1.0):
        """
        data_handler_factory : callable(event_queue) -> DataHandler
        strategy_factory     : callable(data_handler, event_queue) -> Strategy
        """
        self.label = label
        self.event_queue = queue.Queue()
        self.data_handler = data_handler_factory(self.event_queue)
        self.strategy = strategy_factory(self.data_handler, self.event_queue)
        friction = friction or FrictionConfig()
        self.portfolio = Portfolio(self.data_handler, self.event_queue, symbols,
                                    initial_capital=initial_capital, risk_per_trade=risk_per_trade,
                                    max_drawdown_pct=max_drawdown_pct, max_position_pct=max_position_pct,
                                    borrow_cost_bps=friction.borrow_cost_bps,
                                    max_gross_leverage=max_gross_leverage)
        self.execution = SimulatedExecutionHandler(self.event_queue, self.data_handler, friction)
        self.symbols = symbols

    def run(self, progress_callback=None):
        n_total = len(self.data_handler.full_index)
        while self.data_handler.continue_backtest():
            self.data_handler.update_bars()
            self._drain()
            self.portfolio.update_timeindex()
            self._drain()
            if progress_callback:
                progress_callback(self.data_handler._cursor, n_total)
        return self.results()

    def _drain(self):
        while True:
            try:
                event = self.event_queue.get(block=False)
            except queue.Empty:
                break
            self._dispatch(event)

    def _dispatch(self, event):
        if isinstance(event, MarketEvent):
            self.execution.on_market_event(event)
            self.strategy.calculate_signals(event)
        elif isinstance(event, SignalEvent):
            self.portfolio.on_signal(event)
        elif isinstance(event, OrderEvent):
            self.execution.on_order(event)
        elif isinstance(event, FillEvent):
            self.portfolio.on_fill(event)

    def results(self):
        equity = self.portfolio.equity_series()
        trades = pd.DataFrame(self.portfolio.trade_log)
        tca = pd.DataFrame(self.execution.tca_log)
        return {
            "label": self.label,
            "equity": equity,
            "trades": trades,
            "tca": tca,
            "final_equity": equity.iloc[-1] if len(equity) else self.portfolio.initial_capital,
            "n_trades": len(trades),
            "borrow_cost": self.portfolio.total_borrow_cost,
            "halted_at": self.portfolio.halt_timestamp,
        }


def run_arena(configs, progress_callback=None):
    """
    Run several Backtest configurations ("multi-agent") and return a dict
    of label -> results, plus a combined comparison table.

    configs: list of dicts, each with keys matching Backtest.__init__
             (data_handler_factory, strategy_factory, symbols, ...)
    """
    all_results = {}
    for i, cfg in enumerate(configs):
        bt = Backtest(**cfg)
        def _cb(cur, total, i=i, n=len(configs)):
            if progress_callback:
                progress_callback(i, n, cur, total)
        res = bt.run(progress_callback=_cb)
        all_results[res["label"]] = res
    return all_results


PCT_METRICS = {"Total Return", "CAGR", "Annualized Vol", "Max Drawdown", "Alpha (ann.)"}
INT_METRICS = {"# Trades"}
MONEY_METRICS = {"Final Equity", "Total Commission", "Total Slippage Cost", "Total Borrow Cost"}


def format_metric(col: str, v) -> str:
    """Human formatting for a comparison-table cell (amounts are in the data's currency)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    if not isinstance(v, (int, float)):
        return str(v)
    if col in PCT_METRICS or col.startswith("VaR"):
        return f"{v * 100:.2f}%"
    if col in INT_METRICS:
        return f"{int(round(v)):,}"
    if col in MONEY_METRICS:
        return f"{v:,.2f}"
    return f"{v:.2f}"


def build_comparison_table(all_results: dict, benchmark_equity: pd.Series = None,
                            risk_free: float = 0.0, var_confidence: float = 0.95) -> pd.DataFrame:
    rows = {}
    for label, res in all_results.items():
        eq = res["equity"]
        if len(eq) < 2:
            continue
        stats = performance.summarize(eq, benchmark_equity=benchmark_equity, risk_free=risk_free,
                                      var_confidence=var_confidence)
        stats["Final Equity"] = res["final_equity"]
        stats["# Trades"] = res["n_trades"]
        total_commission = res["trades"]["commission"].sum() if len(res["trades"]) else 0.0
        total_slippage = res["trades"]["slippage_cost"].sum() if len(res["trades"]) else 0.0
        stats["Total Commission"] = total_commission
        stats["Total Slippage Cost"] = total_slippage
        stats["Total Borrow Cost"] = res.get("borrow_cost", 0.0)
        rows[label] = stats
    return pd.DataFrame(rows).T.infer_objects()


def export_results(all_results: dict, comparison_table: pd.DataFrame, output_dir: str):
    """Export all backtest results as CSV files to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    comparison_table.to_csv(os.path.join(output_dir, "metrics_summary.csv"))
    for label, res in all_results.items():
        safe = label.replace(" ", "_")
        if len(res["equity"]):
            res["equity"].to_csv(os.path.join(output_dir, f"equity_{safe}.csv"), header=True)
        if len(res["trades"]):
            res["trades"].to_csv(os.path.join(output_dir, f"trade_log_{safe}.csv"), index=False)
        if len(res["tca"]):
            res["tca"].to_csv(os.path.join(output_dir, f"tca_log_{safe}.csv"), index=False)


def export_html_report(all_results: dict, comparison_table: pd.DataFrame,
                        benchmark_equity: pd.Series = None, output_path: str = "report.html"):
    """Generate a self-contained HTML report with inline CSS and embedded data tables."""
    import html as html_lib

    def _fmt(v, col):
        return format_metric(col, v)

    # ---- build metrics table rows ----
    metric_rows = ""
    for label in comparison_table.index:
        row_data = comparison_table.loc[label]
        cells = f'<td class="label">{html_lib.escape(label)}</td>'
        for col in comparison_table.columns:
            val = row_data[col]
            formatted = _fmt(val, col)
            css_class = ""
            if col == "Sharpe":
                css_class = "good" if isinstance(val, (int, float)) and val > 0 else "bad"
            elif col == "Max Drawdown":
                css_class = "bad" if isinstance(val, (int, float)) and val < -0.1 else ""
            elif col == "Total Return":
                css_class = "good" if isinstance(val, (int, float)) and val > 0 else "bad"
            cells += f'<td class="{css_class}">{formatted}</td>'
        metric_rows += f"<tr>{cells}</tr>\n"

    # ---- build trade summary ----
    trade_summaries = ""
    for label, res in all_results.items():
        n = res["n_trades"]
        total_comm = res["trades"]["commission"].sum() if len(res["trades"]) else 0
        total_slip = res["trades"]["slippage_cost"].sum() if len(res["trades"]) else 0
        trade_summaries += (
            f'<div class="card"><h3>{html_lib.escape(label)}</h3>'
            f'<p>{n} trades &middot; {total_comm:,.2f} commission &middot; {total_slip:,.2f} slippage</p>'
            f'<p>Final equity: {res["final_equity"]:,.2f}</p></div>\n'
        )

    # ---- determine winner ----
    if not comparison_table.empty and "Sharpe" in comparison_table.columns:
        winner = comparison_table["Sharpe"].idxmax()
        winner_sharpe = comparison_table.loc[winner, "Sharpe"]
        winner_html = f'<div class="winner">🏆 {html_lib.escape(winner)} — best risk-adjusted return (Sharpe {winner_sharpe:.2f})</div>'
    else:
        winner_html = ""

    metric_headers = "".join(f"<th>{html_lib.escape(c)}</th>" for c in comparison_table.columns)

    report = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Trading Arena Report</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f1117;color:#e6e6e6;padding:32px}}
h1{{color:#4fd1c5;margin-bottom:4px}} h2{{color:#4fd1c5;margin:28px 0 12px;font-size:1.2em}}
.sub{{color:#8892a6;font-size:0.85em;margin-bottom:24px}}
.winner{{background:#1a3a38;border-left:4px solid #4fd1c5;padding:14px 18px;border-radius:6px;margin:18px 0;font-size:1.05em}}
table{{width:100%;border-collapse:collapse;margin:12px 0}}
th{{background:#20242f;color:#4fd1c5;padding:10px 14px;text-align:center;font-size:0.85em}}
td{{padding:8px 14px;text-align:center;border-bottom:1px solid #222;font-size:0.9em}}
td.label{{text-align:left;font-weight:600}}
td.good{{color:#68d391}} td.bad{{color:#fc8181}}
.cards{{display:flex;gap:14px;flex-wrap:wrap}}
.card{{background:#171a23;border:1px solid #2a2f3d;border-radius:8px;padding:16px 20px;min-width:220px;flex:1}}
.card h3{{color:#f6ad55;font-size:1em;margin-bottom:6px}}
.card p{{color:#8892a6;font-size:0.85em;margin:2px 0}}
.footer{{margin-top:40px;color:#555;font-size:0.75em}}
</style></head><body>
<h1>⚔ Trading Strategy Arena — Report</h1>
<p class="sub">Quantitative backtesting results with friction modeling</p>
{winner_html}
<h2>Performance Metrics</h2>
<table><thead><tr><th>Strategy</th>{metric_headers}</tr></thead>
<tbody>{metric_rows}</tbody></table>
<h2>Trade Summary</h2>
<div class="cards">{trade_summaries}</div>
<p class="footer">Generated by Trading Strategy Arena</p>
</body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    return output_path

