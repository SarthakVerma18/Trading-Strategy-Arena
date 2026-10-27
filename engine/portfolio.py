"""
portfolio.py
------------
Converts SignalEvents into OrderEvents (with volatility-scaled position
sizing), applies FillEvents to update cash/positions, and marks the
portfolio to market on every bar to build the equity curve that
performance.py analyzes.

Position sizing: volatility targeting. Each new position is sized so that
its expected contribution to portfolio P&L volatility is roughly constant
across symbols (a $10 vol name and a $500 vol name get very different
share counts for the same "1 unit of risk"). This is standard practice
and prevents a strategy's backtest from being dominated by whichever
symbol happens to have the biggest price.

Orders take `latency_bars` to fill, so the Portfolio tracks in-flight
(sent but not yet filled) quantity per symbol and sizes every new order
against `filled + in-flight` shares. Without this, a signal that arrives
while an earlier order is still pending is sized against a stale
position and the book drifts away from what the strategy intends.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from collections import defaultdict

from .event import OrderEvent

TRADING_DAYS = 252


class Portfolio:
    def __init__(self, data_handler, event_queue, symbols, initial_capital=1_000_000.0,
                 risk_per_trade=0.01, vol_lookback=20, max_drawdown_pct: float = None,
                 max_position_pct: float = None, borrow_cost_bps: float = 0.0,
                 max_gross_leverage: float = 1.0):
        self.data_handler = data_handler
        self.event_queue = event_queue
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade  # target daily P&L stdev per position, as fraction of equity
        self.vol_lookback = vol_lookback
        self.max_drawdown_pct = max_drawdown_pct
        self.max_position_pct = max_position_pct
        self.borrow_cost_bps = borrow_cost_bps  # annualized, charged daily on short market value
        # pre-trade cap on sum(|position notional|) / equity across the whole book
        # (checked when an order is sized; leverage can still drift with prices
        # between trades, exactly as in a real book). None disables.
        self.max_gross_leverage = max_gross_leverage
        self.halted = False
        self.halt_timestamp = None

        self.cash = initial_capital
        self.positions = defaultdict(float)       # symbol -> filled shares (signed)
        self.pending = defaultdict(float)         # symbol -> sent-but-unfilled shares (signed)
        self.avg_entry_price = defaultdict(float)

        self.equity_curve = []   # list of {timestamp, equity}
        self.trade_log = []
        self.realized_pnl = 0.0
        self.total_borrow_cost = 0.0
        self._peak_equity = initial_capital

    # ---------- helpers ----------

    def _last_close(self, sym):
        bars = self.data_handler.get_latest_bars(sym, n=1)
        return bars["Close"].iloc[-1] if len(bars) else None

    def _send_order(self, sym, signed_qty, timestamp, strategy_id, arrival_price):
        if abs(signed_qty) < 1e-6:
            return
        self.pending[sym] += signed_qty
        self.event_queue.put(OrderEvent(
            timestamp=timestamp, symbol=sym, order_type="MKT",
            quantity=abs(signed_qty), direction="BUY" if signed_qty > 0 else "SELL",
            strategy_id=strategy_id, arrival_price=arrival_price,
        ))

    # ---------- signal -> order ----------

    def on_signal(self, signal_event):
        if self.halted:
            return

        sym = signal_event.symbol
        direction = signal_event.direction
        price = self._last_close(sym)
        if price is None or price <= 0:
            return

        target_shares = 0.0
        if direction in ("LONG", "SHORT"):
            hist = self.data_handler.get_full_history_up_to_now(sym)
            rets = hist["Close"].pct_change().tail(self.vol_lookback)
            daily_vol = rets.std()
            if not daily_vol or np.isnan(daily_vol) or daily_vol < 1e-6:
                daily_vol = 0.01
            equity = self._current_equity()
            if signal_event.target_weight is not None:
                shares = equity * abs(signal_event.target_weight) / price
            else:
                risk_dollars = equity * self.risk_per_trade * signal_event.strength
                shares = risk_dollars / (daily_vol * price)

            if self.max_position_pct is not None:
                max_shares = (equity * self.max_position_pct) / price
                shares = min(shares, max_shares)

            if self.max_gross_leverage is not None:
                # only the room left after every OTHER position (filled + in-flight)
                other_gross = 0.0
                for s in set(self.positions) | set(self.pending):
                    if s == sym:
                        continue
                    exposure = self.positions[s] + self.pending[s]
                    if exposure != 0:
                        other_gross += abs(exposure) * (self._last_close(s) or 0.0)
                room = max(self.max_gross_leverage * equity - other_gross, 0.0)
                shares = min(shares, room / price)

            target_shares = shares if direction == "LONG" else -shares

        # size against filled + in-flight shares, never a stale position
        delta = target_shares - (self.positions[sym] + self.pending[sym])
        self._send_order(sym, delta, signal_event.timestamp, signal_event.strategy_id, price)

    # ---------- fill -> position/cash update ----------

    def on_fill(self, fill_event):
        sym = fill_event.symbol
        signed_qty = fill_event.quantity if fill_event.direction == "BUY" else -fill_event.quantity
        self.pending[sym] -= signed_qty
        if abs(self.pending[sym]) < 1e-6:
            self.pending[sym] = 0.0

        prev_shares = self.positions[sym]
        prev_avg = self.avg_entry_price[sym]

        # realized P&L when reducing/closing/flipping a position
        if prev_shares != 0 and np.sign(signed_qty) != np.sign(prev_shares):
            closing_qty = min(abs(signed_qty), abs(prev_shares))
            pnl = closing_qty * (fill_event.fill_price - prev_avg) * np.sign(prev_shares)
            self.realized_pnl += pnl

        new_shares = prev_shares + signed_qty
        if abs(new_shares) < 1e-9:
            new_shares = 0.0
            self.avg_entry_price[sym] = 0.0
        elif np.sign(new_shares) != np.sign(prev_shares) or prev_shares == 0:
            self.avg_entry_price[sym] = fill_event.fill_price
        elif abs(new_shares) > abs(prev_shares):
            total_cost = prev_avg * abs(prev_shares) + fill_event.fill_price * abs(signed_qty)
            self.avg_entry_price[sym] = total_cost / abs(new_shares)
        # (a partial reduction keeps the existing average entry price)

        self.positions[sym] = new_shares
        cash_flow = -signed_qty * fill_event.fill_price
        self.cash += cash_flow - fill_event.commission

        self.trade_log.append({
            "timestamp": fill_event.timestamp, "symbol": sym,
            "direction": fill_event.direction, "quantity": fill_event.quantity,
            "fill_price": fill_event.fill_price, "commission": fill_event.commission,
            "slippage_cost": fill_event.slippage_cost, "strategy_id": fill_event.strategy_id,
        })

    # ---------- mark to market ----------

    def _current_equity(self):
        equity = self.cash
        for sym, shares in self.positions.items():
            if shares == 0:
                continue
            price = self._last_close(sym)
            if price is not None:
                equity += shares * price
        return equity

    def _accrue_borrow(self):
        if self.borrow_cost_bps <= 0:
            return
        for sym, shares in self.positions.items():
            if shares < 0:
                price = self._last_close(sym)
                if price is not None:
                    cost = -shares * price * self.borrow_cost_bps / 1e4 / TRADING_DAYS
                    self.cash -= cost
                    self.total_borrow_cost += cost

    def update_timeindex(self, market_event=None):
        """Mark to market at the bar's close (called after all of the bar's fills)."""
        self._accrue_borrow()
        ts = self.data_handler.current_timestamp()
        equity = self._current_equity()
        self.equity_curve.append({"timestamp": ts, "equity": equity})
        self._peak_equity = max(self._peak_equity, equity)

        if self.max_drawdown_pct is not None and not self.halted and self._peak_equity > 0:
            drawdown = (self._peak_equity - equity) / self._peak_equity
            if drawdown > self.max_drawdown_pct:
                self.halted = True
                self.halt_timestamp = ts
                for sym in list(self.positions.keys()):
                    exposure = self.positions[sym] + self.pending[sym]
                    self._send_order(sym, -exposure, ts, "RISK_LIQUIDATION", self._last_close(sym))

    def equity_series(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype=float)
        df = pd.DataFrame(self.equity_curve).drop_duplicates(subset="timestamp", keep="last")
        df = df.set_index("timestamp").sort_index()
        return df["equity"]
