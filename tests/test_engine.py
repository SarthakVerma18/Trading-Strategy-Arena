"""
Verification suite for the Trading Strategy Arena.

Each section maps to a claim the project makes:
  * event-driven engine benchmarking multiple agents
  * market frictions (slippage, TCA, execution latency) with no look-ahead bias
  * StatArb and multi-factor ML signals evaluated on Sharpe, Alpha/Beta, rolling VaR

Run:  python -m pytest -q
"""

import os
import queue
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import (  # noqa: E402
    Backtest, FrictionConfig, HistoricCSVDataHandler, SyntheticMarketData,
    StatArbStrategy, MLFactorStrategy, BuyAndHoldStrategy, MomentumStrategy,
    MeanReversionStrategy, SimulatedExecutionHandler, OrderEvent, performance,
    run_arena, build_comparison_table, export_results, export_html_report, sensitivity,
)
from engine.data_handler import _BaseBarHandler  # noqa: E402
from engine.strategy import Strategy  # noqa: E402

CSV_DIR = os.path.join(ROOT, "data", "nse_real")
SYMS = ["ICICIBANK", "AXISBANK", "TCS", "INFY", "SBIN", "WIPRO"]


# ---------------------------------------------------------------- helpers

class InMemoryData(_BaseBarHandler):
    """DataHandler over a dict of OHLCV frames (copied, so callers can reuse them)."""

    def __init__(self, event_queue, frames):
        super().__init__(event_queue, {s: df.copy() for s, df in frames.items()})


@pytest.fixture(scope="module")
def real_frames():
    dh = HistoricCSVDataHandler(queue.Queue(), csv_dir=CSV_DIR, symbol_list=SYMS)
    return {s: dh.symbol_data[s] for s in SYMS}


def flat_frames(n=60, price=100.0, symbols=("X",), volume=1_000_000):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return {s: pd.DataFrame({"Open": price, "High": price, "Low": price, "Close": price,
                             "Volume": volume}, index=idx) for s in symbols}


class ScriptedStrategy(Strategy):
    """Emits a pre-defined {bar_number: [(symbol, direction), ...]} schedule."""
    strategy_id = "Scripted"

    def __init__(self, dh, q, schedule):
        super().__init__(dh, q)
        self.schedule = schedule
        self.bar = 0

    def calculate_signals(self, market_event):
        self.bar += 1
        for sym, direction in self.schedule.get(self.bar, []):
            self._emit(sym, direction, market_event.timestamp)


def run(frames, strategy_factory, symbols, friction=None, record_orders=False, **kw):
    bt = Backtest(lambda q: InMemoryData(q, frames), strategy_factory, list(symbols),
                  friction=friction or FrictionConfig(), **kw)
    if record_orders:
        bt.orders = []
        orig = bt.execution.on_order
        def on_order(o):
            # stamp with the engine clock (when the decision was made), not the
            # order's own timestamp, which a leaky strategy could set from the future
            bt.orders.append((bt.data_handler.current_timestamp(), o.symbol, o.direction, round(o.quantity, 6)))
            orig(o)
        bt.execution.on_order = on_order
    return bt, bt.run()


STRATEGY_FACTORIES = {
    "StatArb": lambda dh, q: StatArbStrategy(dh, q, pair=("ICICIBANK", "AXISBANK"), lookback=60),
    "MLFactor": lambda dh, q: MLFactorStrategy(dh, q, symbols=["TCS", "INFY"], train_window=120, refit_every=20),
    "BuyAndHold": lambda dh, q: BuyAndHoldStrategy(dh, q, symbols=SYMS),
    "Momentum": lambda dh, q: MomentumStrategy(dh, q, symbols=SYMS, lookback=126, top_n=2, rebalance_every=21),
    "MeanReversion": lambda dh, q: MeanReversionStrategy(dh, q, symbols=SYMS),
}


# ---------------------------------------------------------------- data integrity

ALL_NSE = ["AXISBANK", "HDFCBANK", "ICICIBANK", "INFY", "ITC", "RELIANCE", "SBIN", "TATAMOTORS", "TCS", "WIPRO"]


@pytest.mark.parametrize("sym", ALL_NSE)
def test_bundled_data_is_split_adjusted(sym):
    """An unadjusted 5:1 split looks like an -80% day and corrupts every backtest."""
    df = pd.read_csv(os.path.join(CSV_DIR, f"{sym}.csv"), index_col=0, parse_dates=True)
    assert not df.isna().any().any()
    assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-6).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-6).all()
    assert df["close"].pct_change().abs().max() < 0.30


@pytest.mark.parametrize("sym", ALL_NSE)
def test_adjustment_preserves_every_non_ex_date_return(sym):
    adj = pd.read_csv(os.path.join(CSV_DIR, f"{sym}.csv"), index_col=0, parse_dates=True)["close"]
    raw = pd.read_csv(os.path.join(ROOT, "data", "nse_raw_unadjusted", f"{sym}.csv"),
                      index_col=0, parse_dates=True)["close"]
    actions = pd.read_csv(os.path.join(ROOT, "data", "corporate_actions.csv"), parse_dates=["ex_date"])
    ex_dates = set(actions.loc[actions.symbol == sym, "ex_date"])
    keep = ~adj.index.isin(ex_dates)
    assert np.allclose(adj.pct_change()[keep].dropna(), raw.pct_change()[keep].dropna(), atol=1e-4)
    assert adj.iloc[-1] == raw.iloc[-1]


# ---------------------------------------------------------------- look-ahead bias

@pytest.mark.parametrize("cut", [441, 504])  # multiples of the 21-bar momentum rebalance
@pytest.mark.parametrize("name", [n for n in STRATEGY_FACTORIES if n != "BuyAndHold"])
def test_no_lookahead_future_data_cannot_change_the_past(real_frames, name, cut):
    """
    The decisive look-ahead test: corrupt every bar from T on with garbage
    prices. If any component (strategy, ML labels, sizing) peeked at the
    future, some decision taken BEFORE T would change. So every order sent
    before T, and the equity curve before T, must be bit-for-bit identical.
    """
    n = 700
    clean = {s: df.iloc[:n] for s, df in real_frames.items()}
    rng = np.random.default_rng(0)
    poisoned = {}
    for s, df in clean.items():
        df = df.copy()
        scale = rng.uniform(0.3, 3.0, n - cut)
        for col in ["Open", "High", "Low", "Close"]:
            df.iloc[cut:, df.columns.get_loc(col)] = df[col].iloc[cut:].values * scale
        df.iloc[cut:, df.columns.get_loc("Volume")] = 1.0
        poisoned[s] = df

    common = dict(symbols=SYMS, friction=FrictionConfig(latency_bars=1), max_position_pct=0.2,
                  record_orders=True)
    bt_c, r_clean = run(clean, STRATEGY_FACTORIES[name], **common)
    bt_p, r_poison = run(poisoned, STRATEGY_FACTORIES[name], **common)

    t_cut = clean["TCS"].index[cut]
    orders_c = [o for o in bt_c.orders if o[0] < t_cut]
    orders_p = [o for o in bt_p.orders if o[0] < t_cut]
    assert len(orders_c) > 0
    assert orders_c == orders_p

    eq_c, eq_p = r_clean["equity"].iloc[:cut], r_poison["equity"].iloc[:cut]
    assert len(eq_c) == cut
    pd.testing.assert_series_equal(eq_c, eq_p)
    # and the corrupted future really did change the later results
    assert not r_clean["equity"].iloc[cut + 5:].equals(r_poison["equity"].iloc[cut + 5:])


def test_strategies_trade_on_real_data(real_frames):
    frames = {s: df.iloc[:700] for s, df in real_frames.items()}
    for name, f in STRATEGY_FACTORIES.items():
        _, r = run(frames, f, SYMS, max_position_pct=0.2)
        assert r["n_trades"] > 0, f"{name} never traded"


def test_late_listing_symbol_is_not_backfilled():
    idx = pd.bdate_range("2020-01-01", periods=30)
    a = flat_frames(30)["X"]
    b = a.iloc[10:].copy() * 2
    dh = InMemoryData(queue.Queue(), {"A": a, "B": b})
    assert dh.full_index[0] == idx[10]
    assert not dh.symbol_data["B"].isna().any().any()


# ---------------------------------------------------------------- execution latency

@pytest.mark.parametrize("latency", [1, 2, 3])
def test_fill_happens_exactly_latency_bars_after_signal(real_frames, latency):
    frames = {s: df.iloc[:400] for s, df in real_frames.items()}
    _, r = run(frames, STRATEGY_FACTORIES["MeanReversion"], SYMS,
               friction=FrictionConfig(latency_bars=latency))
    tca = r["tca"]
    assert len(tca) > 10
    idx = frames["TCS"].index
    lag = idx.get_indexer(tca["timestamp"]) - idx.get_indexer(tca["order_timestamp"])
    assert (lag == latency).all()
    # fills reference the OPEN of the fill bar; arrival price is the signal bar's CLOSE
    for _, row in tca.head(20).iterrows():
        assert row["decision_price"] == frames[row["symbol"]].loc[row["timestamp"], "Open"]
        assert row["arrival_price"] == frames[row["symbol"]].loc[row["order_timestamp"], "Close"]


def test_zero_latency_is_naive_same_bar_close_fill(real_frames):
    frames = {s: df.iloc[:400] for s, df in real_frames.items()}
    _, r = run(frames, STRATEGY_FACTORIES["MeanReversion"], SYMS, friction=FrictionConfig(latency_bars=0))
    tca = r["tca"]
    assert (tca["timestamp"] == tca["order_timestamp"]).all()
    assert np.allclose(tca["shortfall_bps"], tca["slip_bps"])


def test_slippage_model_uses_only_pre_open_information():
    """Changing the fill bar's close/volume must not change the fill price."""
    base = flat_frames(40)["X"]
    base["Close"] = 100 + np.sin(np.arange(40))  # some volatility
    fills = []
    for bump in [1.0, 5.0]:
        df = base.copy()
        df.iloc[30, df.columns.get_loc("Close")] *= bump
        df.iloc[30, df.columns.get_loc("Volume")] *= bump
        q = queue.Queue()
        dh = InMemoryData(q, {"X": df})
        ex = SimulatedExecutionHandler(q, dh, FrictionConfig(latency_bars=1, impact_coefficient=1.0))
        for _ in range(30):
            dh.update_bars()
        ex.on_order(OrderEvent(dh.current_timestamp(), "X", "MKT", 50_000, "BUY"))
        dh.update_bars()
        ex.on_market_event(None)
        fills.append(ex.tca_log[-1]["fill_price"])
    assert fills[0] == fills[1]


# ---------------------------------------------------------------- slippage / TCA

def test_slippage_is_adverse_and_at_least_the_spread(real_frames):
    frames = {s: df.iloc[:500] for s, df in real_frames.items()}
    fr = FrictionConfig(spread_bps=2.0, impact_coefficient=0.1, commission_bps=1.0, commission_min=1.0)
    _, r = run(frames, STRATEGY_FACTORIES["MeanReversion"], SYMS, friction=fr)
    tca = r["tca"]
    buys, sells = tca[tca.direction == "BUY"], tca[tca.direction == "SELL"]
    assert (buys.fill_price > buys.decision_price).all()
    assert (sells.fill_price < sells.decision_price).all()
    assert (tca.slip_bps >= 2.0 - 1e-9).all()
    notional = tca.fill_price * tca.quantity
    assert np.allclose(tca.commission, np.maximum(notional * 1e-4, 1.0))
    assert np.allclose(tca.slippage_cost, (tca.fill_price - tca.decision_price).abs() * tca.quantity)


def test_bigger_orders_slip_more():
    df = flat_frames(40)["X"]
    df["Close"] = 100 + np.sin(np.arange(40))
    slips = []
    for qty in [1_000, 100_000]:
        q = queue.Queue()
        dh = InMemoryData(q, {"X": df})
        ex = SimulatedExecutionHandler(q, dh, FrictionConfig())
        for _ in range(30):
            dh.update_bars()
        ex.on_order(OrderEvent(dh.current_timestamp(), "X", "MKT", qty, "BUY"))
        dh.update_bars()
        ex.on_market_event(None)
        slips.append(ex.tca_log[-1]["slip_bps"])
    assert slips[1] > slips[0]


def test_frictionless_round_trip_pnl_is_exact():
    df = flat_frames(30)["X"]
    df["Open"] = np.linspace(100, 129, 30)
    df["Close"] = df["Open"] + 0.5
    zero = FrictionConfig(commission_bps=0, commission_min=0, spread_bps=0, impact_coefficient=0)
    bt, r = run({"X": df}, lambda dh, q: ScriptedStrategy(dh, q, {5: [("X", "LONG")], 20: [("X", "EXIT")]}),
                ["X"], friction=zero)
    t = r["trades"]
    assert list(t.direction) == ["BUY", "SELL"]
    qty = t.quantity.iloc[0]
    expected = qty * (df["Open"].iloc[20] - df["Open"].iloc[5])  # filled at bar 6 and bar 21 opens
    assert r["final_equity"] - 1_000_000 == pytest.approx(expected)
    assert bt.portfolio.realized_pnl == pytest.approx(expected)


def test_borrow_cost_accrues_only_on_shorts():
    frames = flat_frames(60, symbols=("X",))
    fr = FrictionConfig(commission_bps=0, commission_min=0, spread_bps=0, impact_coefficient=0,
                        borrow_cost_bps=252.0)  # 1 bp per day
    bt, r = run(frames, lambda dh, q: ScriptedStrategy(dh, q, {1: [("X", "SHORT")], 41: [("X", "EXIT")]}),
                ["X"], friction=fr)
    shares = r["trades"].quantity.iloc[0]
    # short is filled at bar 2 open, covered at bar 42 open -> held at closes of bars 2..41 = 40 days
    expected = shares * 100.0 * 1e-4 * 40
    assert r["borrow_cost"] == pytest.approx(expected)
    assert r["final_equity"] == pytest.approx(1_000_000 - expected)

    _, r_long = run(frames, lambda dh, q: ScriptedStrategy(dh, q, {1: [("X", "LONG")], 41: [("X", "EXIT")]}),
                    ["X"], friction=fr)
    assert r_long["borrow_cost"] == 0.0


# ---------------------------------------------------------------- portfolio integrity

def test_signals_while_orders_in_flight_keep_book_consistent():
    """Flip LONG/SHORT/EXIT faster than the latency: the final book must match intent."""
    frames = flat_frames(40)
    schedule = {1: [("X", "LONG")], 2: [("X", "SHORT")], 3: [("X", "EXIT")], 4: [("X", "LONG")],
                5: [("X", "EXIT")], 10: [("X", "SHORT")]}
    bt, _ = run(frames, lambda dh, q: ScriptedStrategy(dh, q, schedule), ["X"],
                friction=FrictionConfig(latency_bars=3))
    pf = bt.portfolio
    target = -(1_000_000 * 0.01 / (0.01 * 100.0))  # flat prices -> vol floor 1% -> sizing formula
    assert pf.pending["X"] == 0
    assert pf.positions["X"] == pytest.approx(target, rel=0.02)


def test_max_drawdown_circuit_breaker_liquidates_and_halts():
    df = flat_frames(60)["X"]
    prices = np.r_[np.full(10, 100.0), np.linspace(100, 40, 20), np.full(30, 40.0)]
    df["Open"] = df["Close"] = prices
    bt, r = run({"X": df}, lambda dh, q: ScriptedStrategy(dh, q, {1: [("X", "LONG")], 45: [("X", "LONG")]}),
                ["X"], max_drawdown_pct=0.05, max_position_pct=1.0, risk_per_trade=1.0)
    assert r["halted_at"] is not None
    assert bt.portfolio.positions["X"] == 0
    assert r["trades"].strategy_id.iloc[-1] == "RISK_LIQUIDATION"
    assert len(r["trades"]) == 2  # the LONG at bar 45 was ignored after the halt


def test_max_position_cap_is_respected(real_frames):
    frames = {s: df.iloc[:400] for s, df in real_frames.items()}
    bt, r = run(frames, STRATEGY_FACTORIES["BuyAndHold"], SYMS, max_position_pct=0.1, risk_per_trade=1.0)
    first = r["trades"].groupby("symbol").first()
    notional = first.quantity * first.fill_price
    assert (notional <= 1_000_000 * 0.1 * 1.05).all()  # 5% headroom for the overnight gap + slippage


@pytest.mark.parametrize("lev", [0.5, 1.0, 2.0])
def test_gross_leverage_cap_is_a_pre_trade_limit(real_frames, lev):
    """
    The cap is checked when an order is sized: no order may take intended
    gross exposure above the cap (or increase it if market moves already
    pushed it above). Between trades, leverage can drift with prices, as in
    any real book.
    """
    frames = {s: df.iloc[:400] for s, df in real_frames.items()}
    bt = Backtest(lambda q: InMemoryData(q, frames), STRATEGY_FACTORIES["MeanReversion"], SYMS,
                  risk_per_trade=0.05, max_gross_leverage=lev)
    pf = bt.portfolio
    gross = lambda: sum(abs(pf.positions[s] + pf.pending[s]) * pf._last_close(s) for s in SYMS) / pf._current_equity()
    peak_after_order = 0.0
    orig = pf.on_signal
    def on_signal(ev):
        nonlocal peak_after_order
        before = gross()
        orig(ev)
        after = gross()
        assert after <= max(lev, before) + 1e-9
        peak_after_order = max(peak_after_order, after if after > before else 0.0)
    pf.on_signal = on_signal
    bt.run()
    assert peak_after_order > 0.9 * lev  # the cap actually binds in this setup


def test_buy_and_hold_is_equal_weight_and_tracks_benchmark(real_frames):
    frames = {s: df.iloc[:600] for s, df in real_frames.items()}
    _, r = run(frames, STRATEGY_FACTORIES["BuyAndHold"], SYMS)
    first = r["trades"].groupby("symbol").first()
    weights = first.quantity * first.fill_price / 1_000_000
    assert len(weights) == len(SYMS)
    assert np.allclose(weights, 1 / len(SYMS), atol=0.01)
    closes = pd.DataFrame({s: frames[s]["Close"] for s in SYMS})
    bench = (closes / closes.iloc[0]).mean(axis=1)
    _, beta = performance.alpha_beta(performance.returns_from_equity(r["equity"]),
                                     performance.returns_from_equity(bench))
    assert beta == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------- StatArb + ML signals

def test_statarb_legs_are_always_opposite_on_cointegrated_pair():
    q = queue.Queue()
    bt = Backtest(lambda q: SyntheticMarketData(q, ["AAA", "BBB"], pairs=[("AAA", "BBB")], n_days=600, seed=3),
                  lambda dh, q: StatArbStrategy(dh, q, pair=("AAA", "BBB")), ["AAA", "BBB"])
    signals = []
    orig = bt.portfolio.on_signal
    bt.portfolio.on_signal = lambda ev: (signals.append((ev.timestamp, ev.symbol, ev.direction)), orig(ev))
    r = bt.run()
    assert r["n_trades"] > 4
    by_ts = pd.DataFrame(signals, columns=["ts", "sym", "dir"]).groupby("ts")["dir"].apply(sorted)
    for dirs in by_ts:
        assert dirs in (["LONG", "SHORT"], ["EXIT", "EXIT"])


def test_ml_factor_walk_forward_refits(real_frames):
    frames = {s: df.iloc[:500] for s, df in real_frames.items()}
    bt, r = run(frames, STRATEGY_FACTORIES["MLFactor"], SYMS)
    strat = bt.strategy
    assert all(m is not None for m in strat.models.values())
    # refit every 20 bars from the first fit at bar train_window + 30 = 150
    assert strat.last_fit_bar["TCS"] >= 480
    assert r["n_trades"] > 0


# ---------------------------------------------------------------- metrics

def test_sharpe_sortino_vol_cagr_known_values():
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0.0005, 0.01, 2520))
    assert performance.sharpe_ratio(rets) == pytest.approx(np.sqrt(252) * rets.mean() / rets.std(ddof=1))
    assert performance.annualized_vol(rets) == pytest.approx(rets.std(ddof=1) * np.sqrt(252))
    eq = pd.Series(100 * (1.10 ** (np.arange(253) / 252)))  # exactly 10%/yr
    assert performance.cagr(eq) == pytest.approx(0.10, abs=0.001)
    assert performance.sortino_ratio(rets) > performance.sharpe_ratio(rets) * 0.9


def test_alpha_beta_recovers_known_regression():
    rng = np.random.default_rng(2)
    bench = pd.Series(rng.normal(0.0004, 0.012, 1500))
    strat = 0.0002 + 1.7 * bench + pd.Series(rng.normal(0, 1e-6, 1500))
    alpha, beta = performance.alpha_beta(strat, bench)
    assert beta == pytest.approx(1.7, abs=1e-3)
    assert alpha == pytest.approx(0.0002 * 252, abs=1e-4)


def test_rolling_var_historical_and_parametric():
    rng = np.random.default_rng(3)
    rets = pd.Series(rng.normal(0, 0.02, 300))
    hist = performance.rolling_var(rets, window=100, confidence=0.95)
    assert hist.iloc[-1] == pytest.approx(-np.percentile(rets.iloc[-100:], 5))
    from scipy.stats import norm
    par = performance.rolling_var(rets, window=100, confidence=0.99, method="parametric")
    w = rets.iloc[-100:]
    assert par.iloc[-1] == pytest.approx(-(w.mean() + norm.ppf(0.01) * w.std()))
    assert (hist.dropna() >= 0).all() and not np.signbit(hist.dropna()).any()


def test_max_drawdown_known_path():
    eq = pd.Series([100, 120, 90, 130, 65, 140])
    assert performance.max_drawdown(eq) == pytest.approx(-0.5)


# ---------------------------------------------------------------- arena / reporting

def test_multi_agent_arena_table_and_exports(real_frames, tmp_path):
    frames = {s: df.iloc[:500] for s, df in real_frames.items()}
    configs = [dict(data_handler_factory=lambda q: InMemoryData(q, frames), strategy_factory=f,
                    symbols=SYMS, label=name, friction=FrictionConfig(borrow_cost_bps=50))
               for name, f in STRATEGY_FACTORIES.items()]
    results = run_arena(configs)
    assert set(results) == set(STRATEGY_FACTORIES)
    closes = pd.DataFrame({s: frames[s]["Close"] for s in SYMS})
    bench = (closes / closes.iloc[0]).mean(axis=1) * 1_000_000
    table = build_comparison_table(results, benchmark_equity=bench, var_confidence=0.99)
    for col in ["Sharpe", "Alpha (ann.)", "Beta", "VaR 99% (latest, 1d)", "Total Borrow Cost"]:
        assert col in table.columns
        assert table[col].notna().all()
    assert table.loc["BuyAndHold", "Beta"] > 0.5  # long-only basket tracks the benchmark
    export_results(results, table, str(tmp_path))
    html = export_html_report(results, table, bench, str(tmp_path / "r.html"))
    assert (tmp_path / "metrics_summary.csv").exists()
    assert "Trading Strategy Arena" in open(html, encoding="utf-8").read()


def test_sensitivity_sweep(real_frames):
    frames = {s: df.iloc[:400] for s, df in real_frames.items()}
    base = dict(data_handler_factory=lambda q: InMemoryData(q, frames), symbols=SYMS, label="sweep",
                strategy_factory_builder=lambda v: (lambda dh, q: MeanReversionStrategy(dh, q, symbols=SYMS, bb_period=int(v))))
    seen = []
    df = sensitivity.run_sensitivity_sweep(base, "bb_period", [10, 20, 30],
                                           progress_callback=lambda i, n: seen.append(i))
    assert list(df.index) == [10, 20, 30] and seen == [0, 1, 2]
    assert isinstance(sensitivity.interpret_sensitivity(df, "bb_period"), str)
