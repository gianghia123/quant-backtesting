from quant import Engine, Portfolio, Order
import cvxpy as cp
import numpy as np
import pandas as pd


class MeanVar(Portfolio):
    def __init__(
        self,
        date_path: str,
        day_to_reset_portfolio: int = 7,
        omega: float = 0.2
    ) -> None:
        super().__init__(date_path)
        self.day_to_reset_portfolio = day_to_reset_portfolio
        self.omega = omega

    def compute_order(self, ohlcv_data: dict[str, pd.DataFrame], today_price: dict[str, float]) -> list[Order | None]:
        if self.day_to_reset_portfolio != 0:
            if self.day_to_reset_portfolio == 2:
                orders = []
                self.day_to_reset_portfolio -= 1
                for ticker, amount in self.stocks:
                    orders.append(Order(ticker, "sell", amount))
                return orders
            else:
                return []
        for ticker, data in ohlcv_data.items():
            returns = np.log(data['adj_close']) - np.log(data['adj_close'].shift(1))
            data['returns'] = returns
        covariance_matrix = [[0.0 for _ in ohlcv_data] for __ in ohlcv_data]
        for idx_a, ticker_a in enumerate(ohlcv_data.keys()):
            for idx_b, ticker_b in enumerate(ohlcv_data.keys()):
                if covariance_matrix[idx_b][idx_a] is not None:
                    covariance_matrix[idx_a][idx_b] == covariance_matrix[idx_b][idx_a]
                else:
                    covariance_matrix[idx_a][idx_b] = (
                        ohlcv_data[ticker_a]['returns'].cov(ohlcv_data[ticker_b]['returns'])
                    )
        covariance_matrix = np.array(covariance_matrix)
        expected_returns = np.array([ohlcv_data[ticker]['returns'].mean() for ticker in ohlcv_data.keys()])

        x = cp.Variable(len(ohlcv_data.keys()))
        returns = expected_returns.T @ x
        var = cp.quad_form(x, covariance_matrix)
        objective = cp.Maximize(returns - self.omega * var)
        constraints = [cp.sum(x) == 1, x >= 0, x <= 0.4]
        problem = cp.Problem(objective, constraints)
        problem.solve()

        orders = []
        for idx, ticker in enumerate(ohlcv_data.keys()):
            allocated_amount = (self.current_cash * 0.1) * x.value[idx]
            # Amount of stocks to buy. Round down to the nearest 100.
            buy_amount = (allocated_amount / today_price[ticker])
            rounded_buy_amount = buy_amount - (buy_amount % 100)
            if rounded_buy_amount <= 0:
                continue
            else:
                orders.append(Order(ticker, "buy", rounded_buy_amount))
        return orders