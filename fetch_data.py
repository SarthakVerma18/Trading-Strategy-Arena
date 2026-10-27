#!/usr/bin/env python3
"""
fetch_data.py
-------------
Fetches REAL historical OHLCV data via the `yfinance` library and caches
it as CSVs under data/<SYMBOL>.csv, in the exact format
HistoricCSVDataHandler expects. Run this once locally (requires internet
access to Yahoo Finance) and the arena will use real data from then on.

Usage:
    pip install yfinance
    python fetch_data.py --symbols AAPL MSFT GOOGL --start 2018-01-01 --end 2024-01-01
    python fetch_data.py --symbols KO PEP --start 2015-01-01           # a classic cointegrated pair
    python fetch_data.py --symbols XLE XOM CVX --period 5y             # sector peers for StatArb

Then in the GUI, keep Data Source = "real", point "CSV dir" at the output folder and enter the same
symbols, or point HistoricCSVDataHandler at csv_dir="data".
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Fetch real market data via yfinance and cache as CSVs.")
    parser.add_argument("--symbols", nargs="+", required=True, help="Ticker symbols, e.g. AAPL MSFT GOOGL")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (overrides --period)")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (used with --start)")
    parser.add_argument("--period", default="3y", help="Lookback period if --start not given, e.g. 1y, 5y, max")
    parser.add_argument("--interval", default="1d", help="Bar interval: 1d, 1wk, 1mo, etc.")
    parser.add_argument("--out", default="data", help="Output directory for CSVs")
    args = parser.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        print("yfinance is not installed. Run:  pip install yfinance", file=sys.stderr)
        sys.exit(1)

    import pandas as pd

    os.makedirs(args.out, exist_ok=True)
    print(f"Fetching {len(args.symbols)} symbol(s) via yfinance...")

    for sym in args.symbols:
        kwargs = dict(interval=args.interval, auto_adjust=True, progress=False)
        if args.start:
            kwargs["start"] = args.start
            kwargs["end"] = args.end
        else:
            kwargs["period"] = args.period

        print(f"  {sym} ...", end=" ", flush=True)
        df = yf.download(sym, **kwargs)
        if df is None or df.empty:
            print("FAILED (no data returned -- check ticker / internet access)")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.index.name = "date"
        df.columns = [c.lower() for c in df.columns]

        out_path = os.path.join(args.out, f"{sym}.csv")
        df.to_csv(out_path)
        print(f"OK -- {len(df)} bars saved to {out_path}")

    print("\nDone. In the GUI, keep Data Source = 'real', set 'CSV dir' to the --out folder and enter these symbols.")


if __name__ == "__main__":
    main()
