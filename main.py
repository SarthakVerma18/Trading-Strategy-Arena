#!/usr/bin/env python3
"""
main.py
-------
Entry point for the Trading Strategy Arena.

  python main.py            -> launches the Tkinter GUI
  python main.py --headless -> runs a default StatArb vs. ML Factor arena
                                in the terminal and prints the comparison
                                table (useful for CI / servers without a
                                display, or a quick smoke test).
"""

import argparse
import sys


def run_headless():
    import queue
    import pandas as pd
    from engine import (
        HistoricCSVDataHandler, StatArbStrategy, MLFactorStrategy,
        BuyAndHoldStrategy, MomentumStrategy, MeanReversionStrategy,
        Backtest, FrictionConfig, run_arena, build_comparison_table, performance,
    )

    # Uses the bundled REAL historical data (NSE large-caps, 2011-2021).
    # ICICIBANK/AXISBANK are correlated bank-sector peers (StatArb pair);
    # TCS/INFY are IT-sector names for the ML factor strategy.
    symbols = ["ICICIBANK", "AXISBANK", "SBIN", "HDFCBANK", "TCS", "INFY", "WIPRO"]
    csv_dir = "data/nse_real"
    friction = FrictionConfig(commission_bps=1.0, spread_bps=2.0, impact_coefficient=0.1, latency_bars=1)

    def data_factory(q):
        return HistoricCSVDataHandler(q, csv_dir=csv_dir, symbol_list=symbols)

    def statarb_factory(dh, q):
        return StatArbStrategy(dh, q, pair=("ICICIBANK", "AXISBANK"), lookback=60, entry_z=2.0)

    def ml_factory(dh, q):
        return MLFactorStrategy(dh, q, symbols=["TCS", "INFY"], train_window=150, refit_every=20)

    def bah_factory(dh, q):
        return BuyAndHoldStrategy(dh, q, symbols=symbols)

    def momentum_factory(dh, q):
        return MomentumStrategy(dh, q, symbols=symbols, lookback=252, top_n=2, rebalance_every=21)

    def mr_factory(dh, q):
        return MeanReversionStrategy(dh, q, symbols=symbols, bb_period=20, bb_std=2.0)

    configs = [
        dict(data_handler_factory=data_factory, strategy_factory=statarb_factory,
             symbols=symbols, initial_capital=1_000_000, friction=friction, label="StatArb",
             max_position_pct=0.15, max_drawdown_pct=0.25),
        dict(data_handler_factory=data_factory, strategy_factory=ml_factory,
             symbols=symbols, initial_capital=1_000_000, friction=friction, label="MLFactor",
             max_position_pct=0.15, max_drawdown_pct=0.25),
        dict(data_handler_factory=data_factory, strategy_factory=bah_factory,
             symbols=symbols, initial_capital=1_000_000, friction=friction, label="BuyAndHold",
             max_position_pct=0.15, max_drawdown_pct=0.25),
        dict(data_handler_factory=data_factory, strategy_factory=momentum_factory,
             symbols=symbols, initial_capital=1_000_000, friction=friction, label="Momentum",
             max_position_pct=0.15, max_drawdown_pct=0.25),
        dict(data_handler_factory=data_factory, strategy_factory=mr_factory,
             symbols=symbols, initial_capital=1_000_000, friction=friction, label="MeanReversion",
             max_position_pct=0.15, max_drawdown_pct=0.25),
    ]

    print("Running Trading Strategy Arena (headless mode, real historical data)...\n")
    results = run_arena(configs)

    bench_dh = HistoricCSVDataHandler(queue.Queue(), csv_dir=csv_dir, symbol_list=symbols)
    closes = pd.DataFrame({s: bench_dh.symbol_data[s]["Close"] for s in symbols})
    bench_equity = (closes / closes.iloc[0]).mean(axis=1) * 1_000_000

    table = build_comparison_table(results, benchmark_equity=bench_equity)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print(table.T.to_string())
    print()
    # the bundled NSE data is priced in INR, so amounts are plain currency units
    for label, res in results.items():
        line = f"{label}: {res['n_trades']} trades, final equity {res['final_equity']:,.2f}"
        if res.get("halted_at") is not None:
            line += f"  (max-drawdown halt on {pd.Timestamp(res['halted_at']).date()})"
        print(line)


def run_gui():
    from gui.app import main as gui_main
    gui_main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading Strategy Arena")
    parser.add_argument("--headless", action="store_true", help="Run without GUI, print results to terminal")
    args = parser.parse_args()

    if args.headless:
        run_headless()
    else:
        try:
            run_gui()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\nGUI failed to launch ({e}). Falling back to --headless mode.\n")
            run_headless()
