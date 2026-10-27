"""
execution.py
------------
Simulated execution handler that turns OrderEvents into FillEvents while
modelling realistic market frictions. This is what makes the backtest
"honest": a naive backtester that fills every order instantly at the
observed close price is systematically optimistic (it implicitly assumes
zero cost, zero market impact, and knowledge of the closing price before
it happens). This handler removes all three privileges:

1. Execution latency
   Orders generated from bar t's signal are not filled at bar t's price;
   they are queued and filled `latency_bars` later (default 1), i.e. at
   the NEXT bar's price. This alone eliminates the single most common
   source of look-ahead bias in amateur backtests.

2. Slippage
   Modelled as a function of the symbol's realized volatility and the
   order's size relative to average volume (a simple square-root market
   impact model), plus a fixed spread cost in basis points. Bigger, more
   urgent orders in more volatile / less liquid names slip more.

3. Transaction Cost Analysis (TCA)
   Every fill records: commission, slippage cost (in $ and bps vs. the
   arrival/decision price), and latency -- so the Portfolio and the GUI
   can report implementation shortfall, not just gross P&L.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import numpy as np

from .event import FillEvent


@dataclass
class FrictionConfig:
    commission_bps: float = 1.0       # fixed cost per trade, in bps of notional
    commission_min: float = 1.0       # minimum $ commission per trade
    spread_bps: float = 2.0           # half-spread cost, in bps of price
    impact_coefficient: float = 0.10  # market impact scaling (sqrt model)
    latency_bars: int = 1             # bars between order and fill (0 = naive same-bar close fill)
    borrow_cost_bps: float = 0.0      # annualized borrow cost for short positions, in bps
                                      # (accrued daily on short market value by the Portfolio)


class SimulatedExecutionHandler:
    """
    Consumes OrderEvents, applies friction, and (after `latency_bars`
    market events have elapsed) emits a FillEvent onto the event queue.

    latency_bars >= 1 : filled at the OPEN of the bar `latency_bars` after
                        the signal bar (the realistic, look-ahead-free setting).
    latency_bars == 0 : filled immediately at the signal bar's CLOSE -- the
                        optimistic "naive backtester" assumption, kept only
                        so its bias can be measured against the realistic one.
    """

    def __init__(self, event_queue, data_handler, friction: FrictionConfig = None):
        self.event_queue = event_queue
        self.data_handler = data_handler
        self.friction = friction or FrictionConfig()
        self._pending = deque()  # orders waiting for latency_bars to elapse
        self.tca_log = []        # list of dicts for post-hoc TCA reporting

    def on_order(self, order_event):
        if self.friction.latency_bars <= 0:
            self._fill_order(order_event, at_close=True)
        else:
            self._pending.append({"order": order_event, "bars_waited": 0})

    def on_market_event(self, market_event):
        """Called every bar; advances pending orders and fills the ones whose latency has elapsed."""
        still_pending = deque()
        while self._pending:
            item = self._pending.popleft()
            item["bars_waited"] += 1
            if item["bars_waited"] >= self.friction.latency_bars:
                self._fill_order(item["order"], at_close=False)
            else:
                still_pending.append(item)
        self._pending = still_pending

    def _fill_order(self, order, at_close: bool):
        sym = order.symbol
        bars = self.data_handler.get_latest_bars(sym, n=1)
        if bars.empty:
            return
        row = bars.iloc[-1]
        hist = self.data_handler.get_full_history_up_to_now(sym)
        if at_close:
            decision_price = row["Close"]
            known = hist                    # the whole bar has printed
        else:
            decision_price = row["Open"]    # fill referenced off the open of the fill bar
            known = hist.iloc[:-1]          # at the open, this bar's close/volume are unknown
        # recent realized volatility (as a fraction) drives both slippage and impact
        rets = known["Close"].pct_change().tail(20)
        vol = rets.std() if len(rets) > 2 and not np.isnan(rets.std()) else 0.01
        avg_volume = known["Volume"].tail(20).mean() if "Volume" in known and len(known) else 1e6
        if not avg_volume or np.isnan(avg_volume):
            avg_volume = 1e6
        participation = min(order.quantity / max(avg_volume, 1.0), 1.0)

        spread_cost_frac = self.friction.spread_bps / 1e4
        impact_frac = self.friction.impact_coefficient * vol * np.sqrt(max(participation, 1e-6))
        total_slip_frac = spread_cost_frac + impact_frac

        sign = 1 if order.direction == "BUY" else -1
        fill_price = decision_price * (1 + sign * total_slip_frac)

        notional = fill_price * order.quantity
        commission = max(notional * self.friction.commission_bps / 1e4, self.friction.commission_min)
        slippage_cost = abs(fill_price - decision_price) * order.quantity

        # implementation shortfall vs. the price seen when the order was decided
        # (captures the overnight gap caused by latency, on top of slippage)
        arrival = order.arrival_price if order.arrival_price else decision_price
        shortfall_bps = sign * (fill_price - arrival) / arrival * 1e4

        fill = FillEvent(
            timestamp=row.name,
            symbol=sym,
            quantity=order.quantity,
            direction=order.direction,
            fill_price=fill_price,
            commission=commission,
            slippage_cost=slippage_cost,
            latency_bars=self.friction.latency_bars,
            strategy_id=order.strategy_id,
        )
        self.tca_log.append({
            "timestamp": row.name, "order_timestamp": order.timestamp,
            "symbol": sym, "direction": order.direction,
            "quantity": order.quantity, "arrival_price": arrival, "decision_price": decision_price,
            "fill_price": fill_price, "slip_bps": total_slip_frac * 1e4,
            "shortfall_bps": shortfall_bps,
            "commission": commission, "slippage_cost": slippage_cost,
            "participation": participation, "volatility": vol,
        })
        self.event_queue.put(fill)
