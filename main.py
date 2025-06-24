import pandas as pd
from utils.data_loader import download_data
from exchange.simulator import ExchangeSimulator
from agents.momentum_agent import MomentumAgent
from backtester.backtest import run_backtest
from backtester.metrics import compute_metrics

df = download_data()
df.to_csv("data/PFE.csv")

df = pd.read_csv("data/PFE.csv", index_col=0, parse_dates=True)
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(inplace=True)


agent = MomentumAgent()
exchange = ExchangeSimulator()

trade_log = run_backtest(agent, df, exchange)
metrics = compute_metrics(trade_log, df)

print(metrics)
