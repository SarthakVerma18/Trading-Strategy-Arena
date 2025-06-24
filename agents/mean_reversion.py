from agents.base_agent import BaseAgent
import numpy as np

class MeanReversionAgent(BaseAgent):
    def __init__(self, window=20, num_std=2):
        super().__init__("MeanReversionAgent")
        self.window = window
        self.num_std = num_std

    def run(self, data):
        rolling_mean = data['Close'].rolling(window=self.window).mean()
        rolling_std = data['Close'].rolling(window=self.window).std()

        upper_band = rolling_mean + (rolling_std * self.num_std)
        lower_band = rolling_mean - (rolling_std * self.num_std)

        signal = np.where(data['Close'] < lower_band, 1, 0)  # long if below lower band
        signal = np.where(data['Close'] > upper_band, -1, signal)  # short if above upper band

        return signal
