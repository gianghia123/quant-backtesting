from quant import Engine, Portfolio, Order
import cvxpy as cp
import numpy as np
import pandas as pd


class MeanVar(Portfolio):
    def __init__(
        self,
        date_path: str,
        day_to_reset_portfolio: int = 7,
        omega: float = 0.2,
        lookback: int = 60
    ) -> None:
        super().__init__(date_path)
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
                allocated = self.current_cash * 0.1 * x.value[idx]
                buy_amount = allocated / today_price[ticker]
                rounded = buy_amount - (buy_amount % 100)
                if rounded > 0:
                    orders.append(Order(ticker, "buy", rounded))
            return orders