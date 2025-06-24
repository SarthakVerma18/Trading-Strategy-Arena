from agents.base_agent import BaseAgent
import numpy as np

class MomentumAgent(BaseAgent):
    def __init__(self, short_window=10, long_window=50):
        super().__init__("MomentumAgent")
        self.short_window = short_window
        self.long_window = long_window

    def run(self, data):
        """
        Generates trading signals based on moving average crossover:
        1 = long, -1 = short, 0 = neutral
        """
        short_ma = data['Close'].rolling(window=self.short_window).mean()
        long_ma = data['Close'].rolling(window=self.long_window).mean()

        signal = np.where(short_ma > long_ma, 1, -1)

        # Optional: if you want to add neutral (0) when moving averages are equal
        signal = np.where(short_ma == long_ma, 0, signal)

        return signal
