from agents.base_agent import BaseAgent
import numpy as np

class ReverseMomentumAgent(BaseAgent):
    def __init__(self, short_window=10, long_window=50):
        super().__init__("ReverseMomentumAgent")
        self.short_window = short_window
        self.long_window = long_window

    def run(self, data):
        short_ma = data['Close'].rolling(window=self.short_window).mean()
        long_ma = data['Close'].rolling(window=self.long_window).mean()

        signal = np.where(short_ma > long_ma, -1, 1)
        signal = np.where(short_ma == long_ma, 0, signal)

        return signal
