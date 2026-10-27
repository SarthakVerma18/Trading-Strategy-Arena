# Data

## `nse_real/` — bundled real historical data (used by default)

Real daily OHLCV for 10 NSE (India) large-cap stocks, 2011-06-29 through
2021-04-30 (2,436 bars each, all on the same dates): ICICIBANK, AXISBANK,
SBIN, HDFCBANK, TCS, INFY, WIPRO, RELIANCE, TATAMOTORS, ITC. Prices are
in INR and **back-adjusted for splits and bonus issues** — no setup
required, works fully offline.

ICICIBANK/AXISBANK (two large private-sector banks, daily-return
correlation ≈ 0.70) are the default StatArb pair. They are not
cointegrated over the whole sample, but their spread is locally
mean-reverting more often than any other pair here, which suits the
strategy's rolling hedge ratio (details in the main README).

## `nse_raw_unadjusted/` and `corporate_actions.csv`

The original, unadjusted files. In these, a split looks like a crash
(e.g. SBIN −90% on 2014-11-20 is its 10:1 split). `corporate_actions.csv`
lists all 14 splits/bonus issues in the window; each factor was checked
against the ex-date price gap and volume jump. To regenerate `nse_real/`:

```bash
python adjust_corporate_actions.py
```

## Fetching your own real data via `yfinance`

Run from the project root:

```bash
pip install yfinance
python fetch_data.py --symbols AAPL MSFT --start 2018-01-01
python fetch_data.py --symbols KO PEP --period 5y --out data/kopep
```

This saves split-adjusted `<SYMBOL>.csv` files to `data/` (or wherever
you pass `--out`), in the same `date,open,high,low,close,volume` format as
`nse_real/`, ready for
`HistoricCSVDataHandler(event_queue, csv_dir="data", symbol_list=[...])`
or the GUI's **CSV dir** field.

The GUI's "yfinance" data source does this fetch live and caches results
to `data/_yf_cache/` automatically — requires internet access to Yahoo
Finance. Delete that folder to force a fresh download.

## Bring your own CSVs

Drop one CSV per symbol in a folder, named `<SYMBOL>.csv`, with columns:
```
date,open,high,low,close,volume
2022-01-03,120.52,121.80,119.10,120.10,3456200
...
```
Prices must be split-adjusted, or splits will be traded as crashes.
