import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .order import Order

DATA_ROOT = Path(__file__).parent.parent / Path("data")

class Portfolio:
    def __init__(
        self,
        date_path: Path | str,
        init_cash: int = 5 * 10 ** 5,
        slippage: float = 0.001,
        commission_rate: float = 0.003,
        tax_rate: float = 0.001,
    ) -> None:
        with open(DATA_ROOT / date_path, 'rb') as f:
            self.date_index = pickle.load(f)
        self.portfolio_value = pd.Series(
            np.zeros(len(self.date_index)),
            index=self.date_index
        )
        self.init_cash = init_cash
        self.current_cash = init_cash
        self.slippage = slippage
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate
        self.stocks = {}

    def compute_order(self, ohlcv_data: dict[str, pd.DataFrame]) -> list[Order]:
        raise NotImplementedError

    def execute_order_t0(self, order: Order) -> None:
        if order.action == 'sell':
            self.stocks[order.ticker] = (
                self.stocks.get(order.ticker, order.amount) - order.amount
            )
        else:
            self.current_cash -= (
                order.price * order.amount *
                (1 + self.slippage + self.commission_rate)
            )

    def execute_order_t2(self, order: Order) -> None:
        if order.action == 'sell':
            self.current_cash += (
                order.price * order.amount *
                (1 - self.slippage - self.commission_rate - self.tax_rate)
            )
        else:
            self.stocks[order.ticker] = self.stocks.get(order.ticker, 0) + order.amount
    
    def update_value(
        self,
        date: pd.Timestamp,
        price_dict: dict[str, float]
    ) -> None:
        self.portfolio_value.loc[date] = self.current_cash
        for ticker, price in price_dict.items():
            amount = self.stocks.get(ticker, 0)
            self.portfolio_value.loc[date] += price * amount

    def analyze(self, pretty_print: bool = False) -> dict[str, float]:
        returns = self.portfolio_value.pct_change().dropna()
        final_return = self.portfolio_value.iloc[-1]
        total_return = (final_return - self.init_cash) / self.init_cash
        cagr = (final_return / self.init_cash) ** (250 / len(self.portfolio_value)) - 1
        sharpe = (returns.mean()) / (returns.std()) * (250 ** 0.5)

        rolling_max = self.portfolio_value.cummax()
        drawdown = (self.portfolio_value - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        return {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown
        }
