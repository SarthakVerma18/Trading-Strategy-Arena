"""
strategy.py
-----------
Two "agents" that the arena benchmarks against each other:

  1. StatArbStrategy   -- classic pairs-trading / statistical arbitrage.
     Trades the z-score of the spread between two cointegrated assets.

  2. MLFactorStrategy   -- multi-factor signal combined via a trained
     classifier (RandomForest) predicting the sign of next-period return
     from momentum, mean-reversion, and volatility factors. Refits
     periodically on a rolling walk-forward window so it never trains on
     future data (no look-ahead).

Both strategies only consume bars that have already been pushed into the
DataHandler's history (via get_full_history_up_to_now), so at bar t they
can see data up to and including t, never t+1.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
import numpy as np
import pandas as pd

from .event import SignalEvent

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


class Strategy(ABC):
    strategy_id: str = "BASE"

    def __init__(self, data_handler, event_queue):
        self.data_handler = data_handler
        self.event_queue = event_queue
        self.positions = {}  # symbol -> 'LONG' / 'SHORT' / 'FLAT'

    @abstractmethod
    def calculate_signals(self, market_event):
        ...

    def _emit(self, symbol, direction, timestamp, strength=1.0, target_weight=None):
        self.event_queue.put(
            SignalEvent(
                strategy_id=self.strategy_id,
                symbol=symbol,
                timestamp=timestamp,
                direction=direction,
                strength=strength,
                target_weight=target_weight,
            )
        )
        self.positions[symbol] = "FLAT" if direction == "EXIT" else direction


class StatArbStrategy(Strategy):
    """
    Pairs-trading statistical arbitrage.

    spread_t = price_A - hedge_ratio * price_B
    z_t      = (spread_t - rolling_mean) / rolling_std

    Entry:  |z| > entry_z   -> short the spread (short A, long B) if z > 0
                               long the spread (long A, short B) if z < 0
    Exit :  |z| < exit_z    -> flatten both legs
    Stop :  |z| > stop_z    -> flatten both legs (risk control)

    The hedge ratio is re-estimated on a rolling OLS window (walk-forward),
    never using data beyond the current bar.
    """

    strategy_id = "StatArb"

    def __init__(self, data_handler, event_queue, pair, lookback=60,
                 entry_z=2.0, exit_z=0.5, stop_z=4.0):
        super().__init__(data_handler, event_queue)
        self.sym_a, self.sym_b = pair
        self.lookback = lookback
        self.entry_z, self.exit_z, self.stop_z = entry_z, exit_z, stop_z
        self.state = "FLAT"  # FLAT, LONG_SPREAD, SHORT_SPREAD

    def calculate_signals(self, market_event):
        hist_a = self.data_handler.get_full_history_up_to_now(self.sym_a)
        hist_b = self.data_handler.get_full_history_up_to_now(self.sym_b)
        if len(hist_a) < self.lookback + 1 or len(hist_b) < self.lookback + 1:
            return

        pa = hist_a["Close"].tail(self.lookback)
        pb = hist_b["Close"].tail(self.lookback)
        # rolling hedge ratio via OLS on the lookback window (walk-forward, no look-ahead)
        beta = np.polyfit(pb.values, pa.values, 1)[0]
        spread = pa.values - beta * pb.values
        mu, sigma = spread.mean(), spread.std(ddof=1)
        if sigma < 1e-8:
            return
        z = (spread[-1] - mu) / sigma
        ts = hist_a.index[-1]

        if self.state == "FLAT":
            if z > self.entry_z:
                self._emit(self.sym_a, "SHORT", ts)
                self._emit(self.sym_b, "LONG", ts)
                self.state = "SHORT_SPREAD"
            elif z < -self.entry_z:
                self._emit(self.sym_a, "LONG", ts)
                self._emit(self.sym_b, "SHORT", ts)
                self.state = "LONG_SPREAD"
        elif self.state == "SHORT_SPREAD":
            if abs(z) < self.exit_z or z > self.stop_z:
                self._emit(self.sym_a, "EXIT", ts)
                self._emit(self.sym_b, "EXIT", ts)
                self.state = "FLAT"
        elif self.state == "LONG_SPREAD":
            if abs(z) < self.exit_z or z < -self.stop_z:
                self._emit(self.sym_a, "EXIT", ts)
                self._emit(self.sym_b, "EXIT", ts)
                self.state = "FLAT"


class MLFactorStrategy(Strategy):
    """
    Multi-factor signal generation combined via a walk-forward-trained
    RandomForestClassifier predicting sign(next 5-day return).

    Factors (computed per symbol, per bar, using only past data):
      - momentum_12_1   : 12-day return, skipping most recent day (classic
                           momentum construction to avoid 1-day reversal)
      - reversal_5      : negative 5-day return (short-term mean reversion)
      - volatility_20   : rolling 20-day realized volatility (risk factor)
      - rsi_14          : relative strength index (overbought/oversold)

    The model is refit every `refit_every` bars on a trailing
    `train_window` of history, and only ever predicts the bar *after*
    the training window ends -- a standard walk-forward setup that
    avoids training on the future.
    """

    strategy_id = "MLFactor"

    def __init__(self, data_handler, event_queue, symbols, train_window=180,
                 refit_every=20, prob_threshold=0.58):
        super().__init__(data_handler, event_queue)
        self.symbols = symbols
        self.train_window = train_window
        self.refit_every = refit_every
        self.prob_threshold = prob_threshold
        self.bar_count = 0
        self.models = {s: None for s in symbols}
        self.last_fit_bar = {s: -10**9 for s in symbols}

    @staticmethod
    def _compute_factors(close: pd.Series) -> pd.DataFrame:
        ret1 = close.pct_change()
        mom_12_1 = close.shift(1) / close.shift(12) - 1.0
        reversal_5 = -(close.pct_change(5))
        vol_20 = ret1.rolling(20).std()
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_14 = 100 - (100 / (1 + rs))
        fwd_ret5 = close.pct_change(5).shift(-5)  # label, only used for TRAINING rows
        df = pd.DataFrame({
            "mom_12_1": mom_12_1,
            "reversal_5": reversal_5,
            "vol_20": vol_20,
            "rsi_14": rsi_14,
            "fwd_ret5": fwd_ret5,
        })
        return df

    def calculate_signals(self, market_event):
        self.bar_count += 1
        if not SKLEARN_OK:
            return

        for sym in self.symbols:
            hist = self.data_handler.get_full_history_up_to_now(sym)
            if len(hist) < self.train_window + 30:
                continue
            # Only look at a bounded trailing window: enough for the rolling
            # factors (20d vol, 14d RSI, train_window for the model) plus a
            # small buffer, never the full history. This keeps per-bar cost
            # O(window) instead of O(len(history)), which matters a lot once
            # the backtest is a few thousand bars long.
            window_len = self.train_window + 40
            close_window = hist["Close"].tail(window_len)
            factors = self._compute_factors(close_window)
            ts = hist.index[-1]

            need_fit = (
                self.models[sym] is None
                or (self.bar_count - self.last_fit_bar[sym]) >= self.refit_every
            )
            if need_fit:
                # Train only on rows whose label (fwd_ret5) is fully known,
                # i.e. everything except the most recent 5 bars.
                train_df = factors.iloc[:-5].dropna()
                train_df = train_df.tail(self.train_window)
                if len(train_df) < 60:
                    continue
                X = train_df[["mom_12_1", "reversal_5", "vol_20", "rsi_14"]].values
                y = (train_df["fwd_ret5"] > 0).astype(int).values
                if len(np.unique(y)) < 2:
                    continue
                clf = RandomForestClassifier(
                    n_estimators=60, max_depth=4, min_samples_leaf=10,
                    random_state=42, n_jobs=1,
                )
                clf.fit(X, y)
                self.models[sym] = clf
                self.last_fit_bar[sym] = self.bar_count

            model = self.models[sym]
            if model is None:
                continue

            latest_row = factors[["mom_12_1", "reversal_5", "vol_20", "rsi_14"]].iloc[[-1]]
            if latest_row.isna().any(axis=None):
                continue
            # Bypass sklearn's joblib-parallel predict_proba dispatch (which
            # has heavy per-call overhead on single-row predictions) and
            # average the trees directly -- mathematically identical result,
            # much faster when called once per bar.
            X = latest_row.values.astype(np.float32)
            tree_probs = np.array([tree.predict_proba(X, check_input=False)[0] for tree in model.estimators_])
            proba_up = tree_probs[:, list(model.classes_).index(1)].mean()

            current = self.positions.get(sym, "FLAT")
            if proba_up > self.prob_threshold and current != "LONG":
                self._emit(sym, "LONG", ts, strength=proba_up)
            elif proba_up < (1 - self.prob_threshold) and current != "SHORT":
                self._emit(sym, "SHORT", ts, strength=1 - proba_up)
            elif (1 - self.prob_threshold) <= proba_up <= self.prob_threshold and current != "FLAT":
                self._emit(sym, "EXIT", ts)


class BuyAndHoldStrategy(Strategy):
    """
    Passive equal-weight benchmark: on the first bar, goes LONG every symbol
    with a target weight of 1/N of equity, then holds forever. It goes
    through the same friction pipeline as every other agent.
    """
    strategy_id = "BuyAndHold"

    def __init__(self, data_handler, event_queue, symbols):
        super().__init__(data_handler, event_queue)
        self.symbols = symbols
        self._entered = False

    def calculate_signals(self, market_event):
        if self._entered:
            return
        for sym in self.symbols:
            if len(self.data_handler.get_latest_bars(sym, n=1)) == 0:
                return
        weight = 1.0 / len(self.symbols)
        for sym in self.symbols:
            self._emit(sym, "LONG", market_event.timestamp, target_weight=weight)
        self._entered = True


class MomentumStrategy(Strategy):
    """
    Classic cross-sectional momentum strategy. 
    Ranks symbols by their trailing `lookback`-day return, skipping the most 
    recent `skip_recent` days to avoid short-term reversal effects.
    Goes LONG the top N symbols, SHORT the bottom N symbols, and exits the rest.
    Rebalances every `rebalance_every` bars.
    """
    strategy_id = "Momentum"

    def __init__(self, data_handler, event_queue, symbols, lookback=252, skip_recent=21, top_n=2, rebalance_every=21):
        super().__init__(data_handler, event_queue)
        self.symbols = symbols
        self.lookback = lookback
        self.skip_recent = skip_recent
        self.top_n = top_n
        self.rebalance_every = rebalance_every
        self.bar_count = 0

    def calculate_signals(self, market_event):
        self.bar_count += 1
        if self.bar_count % self.rebalance_every != 0:
            return

        momenta = {}
        valid_ts = None
        for sym in self.symbols:
            hist = self.data_handler.get_full_history_up_to_now(sym)
            if len(hist) < self.lookback + 1:
                continue
            
            close = hist["Close"].values
            p_now = close[-self.skip_recent - 1]
            p_past = close[-self.lookback - 1]
            
            ret = (p_now / p_past) - 1.0
            momenta[sym] = ret
            valid_ts = hist.index[-1]
            
        if not momenta or len(momenta) < self.top_n * 2:
            return
            
        ranked = sorted(momenta.items(), key=lambda x: x[1])
        shorts = [s for s, m in ranked[:self.top_n]]
        longs = [s for s, m in ranked[-self.top_n:]]
        
        for sym in self.symbols:
            if sym in longs:
                self._emit(sym, "LONG", valid_ts)
            elif sym in shorts:
                self._emit(sym, "SHORT", valid_ts)
            else:
                if self.positions.get(sym, "FLAT") != "FLAT":
                    self._emit(sym, "EXIT", valid_ts)


class MeanReversionStrategy(Strategy):
    """
    Simple single-stock mean reversion using Bollinger Bands.
    Computes a SMA and bands (±std dev). 
    If close < lower band, goes LONG.
    If close > upper band, goes SHORT.
    If close is between SMA ± 0.5 * std dev, EXITS.
    """
    strategy_id = "MeanReversion"

    def __init__(self, data_handler, event_queue, symbols, bb_period=20, bb_std=2.0):
        super().__init__(data_handler, event_queue)
        self.symbols = symbols
        self.bb_period = bb_period
        self.bb_std = bb_std

    def calculate_signals(self, market_event):
        for sym in self.symbols:
            hist = self.data_handler.get_full_history_up_to_now(sym)
            if len(hist) < self.bb_period:
                continue
                
            close = hist["Close"].tail(self.bb_period)
            sma = close.mean()
            std = close.std(ddof=1)
            if std < 1e-8:
                continue
                
            upper_band = sma + self.bb_std * std
            lower_band = sma - self.bb_std * std
            exit_upper = sma + 0.5 * std
            exit_lower = sma - 0.5 * std
            
            curr_price = close.iloc[-1]
            ts = hist.index[-1]
            current_pos = self.positions.get(sym, "FLAT")
            
            if curr_price < lower_band and current_pos != "LONG":
                self._emit(sym, "LONG", ts)
            elif curr_price > upper_band and current_pos != "SHORT":
                self._emit(sym, "SHORT", ts)
            elif exit_lower <= curr_price <= exit_upper and current_pos != "FLAT":
                self._emit(sym, "EXIT", ts)
