"""
data_handler.py
----------------
Supplies OHLCV bars to the backtester one timestamp at a time, bar-by-bar,
so that a Strategy can never see t+1 data while processing t (look-ahead
prevention #1 -- the other half is in execution.py's latency model).

Two concrete handlers are provided:

  * HistoricCSVDataHandler : reads real/user-supplied OHLCV CSVs.
  * SyntheticMarketData    : generates realistic correlated multi-asset
                              price paths (GBM + regime-switching mean
                              reversion) so the engine is fully runnable
                              with zero external data dependencies -- useful
                              for the StatArb pair strategy (needs a
                              cointegrated pair) and the ML factor strategy
                              (needs enough history + cross-sectional
                              variety to learn from).
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .event import MarketEvent


class DataHandler(ABC):
    """Abstract base class -- all data handlers must implement this interface."""

    @abstractmethod
    def get_latest_bars(self, symbol: str, n: int = 1) -> pd.DataFrame:
        ...

    @abstractmethod
    def update_bars(self):
        """Advance the internal clock by one bar and push a MarketEvent."""
        ...

    @abstractmethod
    def continue_backtest(self) -> bool:
        ...


class _BaseBarHandler(DataHandler):
    """Shared bookkeeping for handlers that hold a dict[symbol -> DataFrame]."""

    def __init__(self, event_queue, symbol_data: Dict[str, pd.DataFrame]):
        self.event_queue = event_queue
        self.symbol_data = symbol_data
        self.symbols: List[str] = list(symbol_data.keys())

        # Align every symbol on the same DatetimeIndex (outer join, ffill).
        # The index starts at the first date on which EVERY symbol has data:
        # back-filling a late-listing symbol would leak its future prices
        # into earlier bars (a subtle look-ahead), so those bars are dropped.
        all_index = None
        for df in symbol_data.values():
            all_index = df.index if all_index is None else all_index.union(df.index)
        all_index = all_index.sort_values()
        common_start = max(df.index.min() for df in symbol_data.values())
        self.full_index = all_index[all_index >= common_start]

        for sym in self.symbols:
            self.symbol_data[sym] = (
                self.symbol_data[sym]
                .reindex(self.full_index)
                .ffill()
            )

        self._cursor = 0  # index into full_index
        self.continue_flag = True

    def get_latest_bars(self, symbol: str, n: int = 1) -> pd.DataFrame:
        return self.symbol_data[symbol].iloc[max(0, self._cursor - n):self._cursor]

    def get_full_history_up_to_now(self, symbol: str) -> pd.DataFrame:
        return self.symbol_data[symbol].iloc[:self._cursor]

    def continue_backtest(self) -> bool:
        return self.continue_flag

    def current_timestamp(self):
        if self._cursor == 0:
            return self.full_index[0]
        return self.full_index[self._cursor - 1]

    def update_bars(self):
        """
        Advance the clock by one bar and push a MarketEvent. Note that the
        full OHLCV history for every symbol is already loaded in memory at
        construction time (see subclasses) -- update_bars() doesn't reveal
        new *data*, it advances the *cursor* that controls how much of that
        already-loaded data get_full_history_up_to_now()/get_latest_bars()
        are allowed to return. This is what prevents a Strategy from seeing
        future bars while keeping the per-bar cost O(1) instead of O(n)
        (an earlier version rebuilt each symbol's revealed history via
        pd.concat every bar, which made an N-bar backtest cost O(N^2)).
        """
        if self._cursor >= len(self.full_index):
            self.continue_flag = False
            return
        ts = self.full_index[self._cursor]
        self._cursor += 1
        self.event_queue.put(MarketEvent(timestamp=ts))
        if self._cursor >= len(self.full_index):
            self.continue_flag = False


class HistoricCSVDataHandler(_BaseBarHandler):
    """
    Reads one CSV per symbol from `csv_dir`, expecting columns:
    date, open, high, low, close, volume  (case-insensitive; 'date' as index)
    """

    def __init__(self, event_queue, csv_dir: str, symbol_list: List[str]):
        symbol_data = {}
        for sym in symbol_list:
            path = os.path.join(csv_dir, f"{sym}.csv")
            df = pd.read_csv(path)
            df.columns = [c.strip().lower() for c in df.columns]
            date_col = "date" if "date" in df.columns else df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col).sort_index()
            rename = {c: c.capitalize() for c in ["open", "high", "low", "close", "volume"] if c in df.columns}
            df = df.rename(columns=rename)
            symbol_data[sym] = df
        super().__init__(event_queue, symbol_data)


class YFinanceDataHandler(_BaseBarHandler):
    """
    Fetches real historical OHLCV data from Yahoo Finance via the
    `yfinance` library -- this is the "fetch real market data through a
    library" path. Requires outbound internet access to
    query1/query2.finance.yahoo.com (works when run locally; sandboxed
    environments with restricted network egress will need to fall back
    to HistoricCSVDataHandler with pre-downloaded CSVs instead).

    Data is fetched once at construction time for the full symbol list,
    then bars are streamed out one at a time via update_bars() exactly
    like every other DataHandler, so strategies can't distinguish it
    from historic CSV or synthetic data.
    """

    def __init__(self, event_queue, symbol_list: List[str], start: str = None,
                 end: str = None, period: str = "3y", interval: str = "1d",
                 cache_dir: Optional[str] = None):
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError(
                "yfinance is required for YFinanceDataHandler. Install it with:\n"
                "    pip install yfinance"
            ) from e

        symbol_data = {}
        for sym in symbol_list:
            cache_path = os.path.join(cache_dir, f"{sym}.csv") if cache_dir else None
            if cache_path and os.path.exists(cache_path):
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            else:
                kwargs = dict(interval=interval, auto_adjust=True, progress=False)
                if start:
                    kwargs["start"] = start
                    kwargs["end"] = end
                else:
                    kwargs["period"] = period
                raw = yf.download(sym, **kwargs)
                if raw is None or raw.empty:
                    raise ValueError(
                        f"yfinance returned no data for '{sym}'. Check the ticker symbol, "
                        f"your internet connection, and that Yahoo Finance is reachable "
                        f"from this network."
                    )
                # yfinance >= 0.2 returns a MultiIndex column frame for single tickers too
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                df = raw.rename(columns={
                    "Open": "Open", "High": "High", "Low": "Low",
                    "Close": "Close", "Volume": "Volume",
                })[["Open", "High", "Low", "Close", "Volume"]]
                if cache_path:
                    os.makedirs(cache_dir, exist_ok=True)
                    df.to_csv(cache_path)
            symbol_data[sym] = df
        super().__init__(event_queue, symbol_data)


class SyntheticMarketData(_BaseBarHandler):
    """
    Generates synthetic daily OHLCV data for a set of symbols with a
    controllable correlation / cointegration structure:

      * 'trend' symbols follow GBM with drift + volatility clustering
        (GARCH-like) -- good for the ML factor strategy.
      * symbols listed in `pairs` are constructed as a cointegrated pair:
        asset B = asset A + stationary mean-reverting spread -- exactly
        the structure a StatArb strategy is designed to exploit.

    This keeps the whole project runnable end-to-end with no network
    access and no API keys, while still producing statistically
    realistic test data (fat-tailed returns, vol clustering, a genuine
    cointegrated relationship for pairs trading).
    """

    def __init__(
        self,
        event_queue,
        symbols: List[str],
        pairs: Optional[List[tuple]] = None,
        n_days: int = 756,          # ~3 trading years
        start: str = "2022-01-03",
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        pairs = pairs or []
        paired_symbols = {s for pair in pairs for s in pair}
        dates = pd.bdate_range(start=start, periods=n_days)

        symbol_data = {}

        # 1) Independent / trend symbols via GARCH(1,1)-like vol clustering + drift
        for sym in symbols:
            if sym in paired_symbols:
                continue
            symbol_data[sym] = self._gen_trend_asset(rng, dates, base_price=rng.uniform(50, 300))

        # 2) Cointegrated pairs: B = A * beta + noise, spread mean-reverts (OU process)
        for (a, b) in pairs:
            base_price = rng.uniform(50, 300)
            df_a = self._gen_trend_asset(rng, dates, base_price=base_price)
            beta = rng.uniform(0.8, 1.3)
            # scale the mean-reverting spread relative to price level (~1-3% of price)
            spread_sigma = base_price * rng.uniform(0.01, 0.03)
            spread = self._gen_ou_process(rng, len(dates), mu=0.0, theta=0.06, sigma=spread_sigma)
            close_b = df_a["Close"] * beta + spread
            df_b = self._ohlcv_from_close(rng, dates, close_b)
            symbol_data[a] = df_a
            symbol_data[b] = df_b

        super().__init__(event_queue, symbol_data)
        self.pairs = pairs

    @staticmethod
    def _gen_ou_process(rng, n, mu, theta, sigma):
        x = np.zeros(n)
        x[0] = mu
        for t in range(1, n):
            x[t] = x[t - 1] + theta * (mu - x[t - 1]) + sigma * rng.standard_normal()
        return x

    def _gen_trend_asset(self, rng, dates, base_price=100.0):
        n = len(dates)
        # GARCH(1,1)-like volatility clustering, calibrated for ~1-2% daily vol:
        # unconditional variance = omega / (1 - alpha - beta)
        omega, alpha, beta = 2e-5, 0.08, 0.88
        vol2 = np.zeros(n)              # tracks variance, not vol, to keep GARCH recursion stable
        vol2[0] = omega / (1 - alpha - beta)
        shocks = rng.standard_normal(n)
        rets = np.zeros(n)
        drift = rng.uniform(-0.0002, 0.0006)
        for t in range(1, n):
            vol2[t] = omega + alpha * rets[t - 1] ** 2 + beta * vol2[t - 1]
            vol_t = np.sqrt(max(vol2[t], 1e-10))
            # occasional regime jump for fat tails (small probability, bounded magnitude)
            jump = np.clip(rng.standard_t(df=4), -4, 4) * 0.006 if rng.random() < 0.015 else 0.0
            rets[t] = drift + vol_t * shocks[t] + jump
        rets = np.clip(rets, -0.15, 0.15)  # sanity bound: no single-day moves beyond +-15%
        close = base_price * np.cumprod(1 + rets)
        return self._ohlcv_from_close(rng, dates, close)

    @staticmethod
    def _ohlcv_from_close(rng, dates, close):
        close = np.asarray(close)
        intraday_vol = np.abs(rng.normal(0.004, 0.002, len(close)))
        high = close * (1 + intraday_vol)
        low = close * (1 - intraday_vol)
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        volume = rng.integers(1_000_000, 8_000_000, len(close))
        df = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=dates,
        )
        return df
