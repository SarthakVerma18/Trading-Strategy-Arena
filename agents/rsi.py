from agents.base_agent import BaseAgent
import numpy as np
import pandas as pd

class RSIAgent(BaseAgent):
    def __init__(self, window=14, lower=30, upper=70):
        super().__init__("RSIAgent")
        self.window = window
        self.lower = lower
        self.upper = upper

    def compute_rsi(self, series):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=self.window).mean()
        avg_loss = loss.rolling(window=self.window).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def run(self, data):
        rsi = self.compute_rsi(data['Close'])
        signal = np.where(rsi < self.lower, 1, 0)
        signal = np.where(rsi > self.upper, -1, signal)
        return signal
