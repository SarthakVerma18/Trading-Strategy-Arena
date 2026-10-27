#!/usr/bin/env python3
"""
adjust_corporate_actions.py
---------------------------
Back-adjusts raw (unadjusted) daily OHLCV CSVs for stock splits and bonus
issues, so a 5:1 split is not mistaken for an 80% crash.

For every event in the actions file, all bars BEFORE the ex-date have
their prices divided by `factor` and their volume multiplied by it. This is
the standard back-adjustment: the most recent prices are unchanged, and
every daily return (and every split-free price ratio) is preserved.

Usage:
    python adjust_corporate_actions.py                       # bundled NSE data
    python adjust_corporate_actions.py --raw data/nse_raw_unadjusted \
        --actions data/corporate_actions.csv --out data/nse_real
"""

import argparse
import glob
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def adjust(df: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    df = df.astype({"volume": "float64"})
    for _, a in actions.iterrows():
        before = df.index < a["ex_date"]
        for col in ["open", "high", "low", "close"]:
            df.loc[before, col] = df.loc[before, col] / a["factor"]
        df.loc[before, "volume"] = df.loc[before, "volume"] * a["factor"]
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].round(4)
    df["volume"] = df["volume"].round().astype("int64")
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    p.add_argument("--raw", default=os.path.join(HERE, "data", "nse_raw_unadjusted"))
    p.add_argument("--actions", default=os.path.join(HERE, "data", "corporate_actions.csv"))
    p.add_argument("--out", default=os.path.join(HERE, "data", "nse_real"))
    args = p.parse_args()

    actions = pd.read_csv(args.actions, parse_dates=["ex_date"])
    os.makedirs(args.out, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(args.raw, "*.csv"))):
        sym = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.columns = [c.strip().lower() for c in df.columns]
        sym_actions = actions[actions["symbol"] == sym]
        out = adjust(df, sym_actions)
        out.index.name = "date"
        out.to_csv(os.path.join(args.out, f"{sym}.csv"))
        worst = out["close"].pct_change().abs().max()
        print(f"{sym:11s} {len(sym_actions)} action(s) applied, largest daily move now {worst:.1%}")


if __name__ == "__main__":
    main()
