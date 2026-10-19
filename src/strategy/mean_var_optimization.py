from pathlib import Path
from quant import Engine, Portfolio, Order
import cvxpy as cp
import numpy as np
import pandas as pd
from data_engine import get_historical_data, get_today_opens

class MeanVar(Portfolio):
    def __init__(
        self,
        date: str | pd.DatetimeIndex,
        day_to_reset_portfolio: int = 7,
        omega: float = 0.2,
        lookback: int = 60
    ) -> None:
        if isinstance(date, str):
            super().__init__(date_path=date)
        elif isinstance(date, pd.DatetimeIndex):
            super().__init__(date_index=date)
        else:
            raise ValueError("WTF")
        self.day_to_reset_portfolio = day_to_reset_portfolio
        self.current_day = day_to_reset_portfolio
        self.omega = omega
        self.lookback = lookback

    def compute_order(self, ohlcv_data: dict[str, pd.DataFrame], today_price: dict[str, float]) -> list[Order | None]:
        if self.current_day != 0:
            if self.current_day == 2:
                orders = []
                self.current_day -= 1
                for ticker, amount in self.stocks.items():
                    orders.append(Order(ticker, "sell", amount))
                return orders
            else:
                self.current_day -= 1
                return []
        else:
            # Build a returns matrix: rows = dates, columns = tickers
            prices = pd.DataFrame({t: d['adj_close'] for t, d in ohlcv_data.items()})
            prices = prices.tail(self.lookback + 1)
            rets = np.log(prices).diff().dropna(how="all")

            # Drop tickers without enough history, fill remaining gaps
            rets = rets.dropna(axis=1, thresh=max(2, int(0.8 * len(rets))))
            rets = rets.fillna(0.0)
            tickers = list(rets.columns)

            if len(rets) < 10 or len(tickers) == 0:   # not enough data to estimate
                self.current_day = self.day_to_reset_portfolio
                return []

            cov = rets.cov().values
            cov = (cov + cov.T) / 2                    # enforce symmetry
            mu = rets.mean().values

            assert np.isfinite(cov).all() and np.isfinite(mu).all()

            x = cp.Variable(len(tickers))
            objective = cp.Maximize(mu @ x - self.omega * cp.quad_form(x, cp.psd_wrap(cov)))
            constraints = [cp.sum(x) == 1, x >= 0, x <= 0.4]
            cp.Problem(objective, constraints).solve()

            self.current_day = self.day_to_reset_portfolio

            if x.value is None:
                return []

            orders = []
            for idx, ticker in enumerate(tickers):
                allocated = self.current_cash * 0.9 * x.value[idx]
                buy_amount = allocated / today_price[ticker]
                rounded = buy_amount - (buy_amount % 100)
                if rounded > 0:
                    orders.append(Order(ticker, "buy", rounded))
            return orders

if __name__ == "__main__":
    save_path = Path("save_mv")
    if save_path.exists():
        with open(save_path, "r") as f:
            current_day = int(f.readline())
    else:
        current_day = 0
    price_data, price_index = get_historical_data()
    price_today = get_today_opens()
    portfolio = MeanVar(price_index, day_to_reset_portfolio=30, omega=2, lookback=60)
    portfolio.current_day = current_day
    orders = portfolio.compute_order(price_data, price_today)
    for order in orders:
        order.date = pd.Timestamp.today()
        order.price = 0.0
    print(orders)
    with open(save_path, "w") as f:
        f.truncate(0)
        print(portfolio.current_day, file=f)
