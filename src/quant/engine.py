import math
import pickle
import warnings
from pathlib import Path

import pandas as pd

from .order import Order
from .portfolio import Portfolio

DELTA_1DAY = pd.Timedelta(1, 'D')
DATA_ROOT = Path(__file__).parent.parent / Path("data")

class Engine:
    def __init__(self, data_path: str | Path) -> None:
        with open(DATA_ROOT / Path(data_path), 'rb') as f:
            self.data = pickle.load(f)
        self.tickers = list(self.data.keys())
        self.last_close = {}
        self.trade_log = list()

    def log(self, order: Order) -> None:
        self.trade_log.append(order)

    def get_trade_log(self) -> pd.DataFrame:
        temp = pd.DataFrame(self.trade_log)
        temp.set_index('date', inplace=True)
        return temp

    def simulate(
            self,
            portfolio: Portfolio,
            start_date: str | pd.Timestamp | None = None,
            end_date: str | pd.Timestamp | None = None
    ) -> None:
        if start_date is not None and end_date is not None:
            start_index, end_index = portfolio.date_index.get_indexer(
                [start_date, end_date], method="nearest"
            )
            date_index = portfolio.date_index[start_index: end_index]
        else:
            date_index = portfolio.date_index
        t2_order: list[Order] = []
        t1_order: list[Order] = []
        for date in date_index:
            # Settles previous T+1 and T+2 orders
            for order in t1_order:
                portfolio.execute_order_t2(order)
            t1_order = list(t2_order)
            t2_order = list()

            # Aggegrate data for inputing to compute_order
            ohlcv = {k: v.loc[:date - DELTA_1DAY] for k, v in self.data.items()}
            price_today = {}
            for ticker in self.tickers:
                price = self.data[ticker]['adj_close'].get(date)
                if pd.notna(price):
                    self.last_close[ticker] = price
                if ticker in self.last_close:
                    price_today[ticker] = price
            orders = portfolio.compute_order(ohlcv, price_today)

            # Takes in today's order
            for order in orders:
                if order is not None:
                    ticker = order.ticker
                    order.date = date
                    price = self.data[ticker]['adj_open'].get(date)
                    if pd.isna(price) or price is None:
                        continue
                    else:
                        order.price = self.data[ticker].loc[date]['adj_open']
                        deducted_cash = (
                            order.price * order.amount *
                            (1 + portfolio.slippage + portfolio.commission_rate)
                        )
                        if (order.action == 'sell' and
                            portfolio.stocks.get(order.ticker, 0) >= order.amount):
                            portfolio.execute_order_t0(order)
                            t2_order.append(order)
                            self.log(order)
                        elif order.action == 'sell':
                            raise ValueError(
                                f"The strategy demands selling "
                                f"{order.amount} of {order.ticker} "
                                f"while only have "
                                f"{portfolio.stocks.get(order.ticker, 0)}."
                            )
                        elif (order.action == 'buy' and
                              portfolio.current_cash >= deducted_cash):
                            portfolio.execute_order_t0(order)
                            t2_order.append(order)
                            self.log(order)
                        elif order.action == 'buy':
                            raise ValueError(
                                f"The strategy demands buying "
                                f"{deducted_cash} worth of stocks "
                                f"while only have {portfolio.current_cash}."
                            )
                        else:
                            raise ValueError(
                                f"Expected 'buy' or 'sell' from order.action, "
                                f"got {order.action}."
                            )

            # Settles today's price
            price_dict = {}
            for ticker in self.tickers:
                price = self.data[ticker]['adj_close'].get(date)
                if pd.notna(price):
                    self.last_close[ticker] = price
                if ticker in self.last_close:
                    price_dict[ticker] = self.last_close[ticker]
                portfolio.update_value(date, price_dict)
