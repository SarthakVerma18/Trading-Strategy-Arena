import yfinance as yf
import pandas as pd
import importlib.util
import sys
from pathlib import Path

def download_data(ticker="PFE", start="2010-01-01", end="2025-04-16"):
    df = yf.download(ticker, start=start, end=end)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    return df

def load_strategy_from_file(filepath):
    module_name = Path(filepath).stem
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and callable(getattr(attr, 'run', None)):
            instance = attr()
            if not hasattr(instance, "name"):
                instance.name = module_name
            return instance

    raise ValueError(f"No valid strategy class with a 'run()' method found in {filepath}")
