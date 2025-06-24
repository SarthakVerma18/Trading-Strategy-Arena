class ExchangeSimulator:
    def __init__(self, initial_cash=1_000_000, slippage_pct=0.001):
        self.cash = initial_cash
        self.position = 0
        self.trade_log = []
        self.slippage_pct = slippage_pct

    def execute_order(self, date, price, size):
        # Apply slippage
        effective_price = price * (1 + self.slippage_pct * (-1 if size > 0 else 1))
        cost = effective_price * size

        # Update cash and position
        self.cash -= cost
        self.position += size

        self.trade_log.append({
            "date": date,
            "price": effective_price,
            "size": size,
            "cash": self.cash,
            "position": self.position
        })

    def get_portfolio_value(self, current_price):
        return self.cash + self.position * current_price
