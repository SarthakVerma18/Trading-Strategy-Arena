# custom_strategies/my_strategy.py

import numpy as np
import pandas as pd

class CustomAgent:
    """
    A simple moving average crossover strategy as a user-uploaded example.
    Long when short MA crosses above long MA.
    Short when short MA crosses below long MA.
    """

    def __init__(self, short_window=10, long_window=50):
        self.short_window = short_window
        self.long_window = long_window

    def run(self, df: pd.DataFrame) -> np.ndarray:
        if 'Close' not in df.columns:
            raise ValueError("Input DataFrame must contain 'Close' column")

        df = df.copy()
        df['SMA_short'] = df['Close'].rolling(self.short_window).mean()
        df['SMA_long'] = df['Close'].rolling(self.long_window).mean()

        signal = np.where(df['SMA_short'] > df['SMA_long'], 1,
                          np.where(df['SMA_short'] < df['SMA_long'], -1, 0))

        return signal
