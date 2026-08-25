import pickle

import numpy as np
import pandas as pd
import pytest

import quant

# Sub-class of Portfolio, with multiple simplification.
# Argument: orders, a list of orders in the form of [{ticker: Order}, ...]
# Each entry in orders correspond to orders executed in that day.
# DO NOT TOUCH THIS!
class FakePortfolio(quant.Portfolio):
    def __init__(self, orders: list[dict[str, quant.Order]]):
        super().__init__(
            date_path="trading_date.pickle",
            commission_rate=0.0,
            slippage=0.0,
            tax_rate=0.0
        )
        self.date_index = pd.date_range("2020-01-04", "2020-01-07")
        self.portfolio_value = pd.Series(
            np.zeros(len(self.date_index)),
            index=self.date_index
        ) 
        self.day_index = 0
        self.orders = orders
        self.t0_order = []
        self.t2_order = []

    def compute_order(
            self,
            ticker: str,
            ohlcv_data: pd.DataFrame
    ) -> quant.Order | None:
        indx = len(ohlcv_data)
        try:
            temp = self.orders[indx].get(ticker)
            return temp
        except IndexError:
            return None

    def execute_order_t0(self, order: quant.Order) -> None:
        self.t0_order.append((self.day_index, order))
        super().execute_order_t0(order)

    def execute_order_t2(self, order: quant.Order) -> None:
        self.t2_order.append((self.day_index, order))
        super().execute_order_t2(order)

    def update_value(self, date: pd.Timestamp, price_dict: dict[str, float]) -> None:
        super().update_value(date, price_dict)
        self.day_index += 1

# Setting up Engine and data to test. The data must be 4 days long, with 2
# tickers, and constant price (do not use random data).

@pytest.fixture
def engine(tmp_path):
    """Engine with 4 days of constant-price data for 2 tickers (AAA, BBB)."""
    dates = pd.date_range("2020-01-04", "2020-01-07")
    data = {
        "AAA": pd.DataFrame(
            {"adj_open": [10.0] * 4, "adj_close": [10.0] * 4},
            index=dates,
        ),
        "BBB": pd.DataFrame(
            {"adj_open": [10.0] * 4, "adj_close": [10.0] * 4},
            index=dates,
        ),
    }
    data_path = tmp_path / "data.pickle"
    with open(data_path, "wb") as f:
        pickle.dump(data, f)
    return quant.Engine(data_path)


# Testing whether the orders settles 2 days after placement.
# Pass 1 order in day 1 of simulation, then check if the date index in t0_order
# and t2_order are exactly 2 days apart.

def test_order_settles_2_days_after_placement(engine):
    order = quant.Order(
        ticker="AAA",
        action="buy",
        amount=10,
        date=pd.Timestamp("2020-01-04"),
        price=10.0,
    )
    # 1 order on day 1 of the simulation.
    portfolio = FakePortfolio([{"AAA": order}])
    engine.simulate(portfolio)

    assert len(portfolio.t0_order) == 1
    assert len(portfolio.t2_order) == 1

    t0_day, t0_order = portfolio.t0_order[0]
    t2_day, t2_order = portfolio.t2_order[0]
    assert t2_day - t0_day == 2
    assert t2_order is t0_order


# Testing if buying and selling is normal.

def test_buying_and_selling_is_normal(engine):
    buy_order = quant.Order(
        ticker="AAA",
        action="buy",
        amount=50,
        date=pd.Timestamp("2020-01-04"),
        price=10.0,
    )
    sell_order = quant.Order(
        ticker="AAA",
        action="sell",
        amount=30,
        date=pd.Timestamp("2020-01-05"),
        price=10.0,
    )
    portfolio = FakePortfolio([{"AAA": buy_order}, {"AAA": sell_order}])
    portfolio.stocks = {"AAA": 100}
    engine.simulate(portfolio)

    # Both orders are executed on the day they are placed...
    assert [(day, o) for day, o in portfolio.t0_order] == [
        (0, buy_order),
        (1, sell_order),
    ]
    # ...and settle exactly 2 days later.
    assert [(day, o) for day, o in portfolio.t2_order] == [
        (2, buy_order),
        (3, sell_order),
    ]

    # The engine fills in the execution date and open price.
    assert buy_order.date == pd.Timestamp("2020-01-04")
    assert sell_order.date == pd.Timestamp("2020-01-05")
    assert buy_order.price == 10.0
    assert sell_order.price == 10.0

    # Both executed orders are logged.
    assert engine.trade_log == [buy_order, sell_order]

    # Correct T+2 bookkeeping: the buy settles into stocks, the sell into cash.
    assert portfolio.current_cash == 499800  # 500000 - 50*10 (buy) + 30*10 (sell)
    assert portfolio.stocks == {"AAA": 120}  # 100 - 30 (sell) + 50 (buy)
    assert portfolio.portfolio_value.tolist() == [
        500500.0,  # day 1: cash 499500 + 100 shares
        500200.0,  # day 2: cash 499500 + 70 shares
        500700.0,  # day 3: cash 499500 + 120 shares
        501000.0,  # day 4: cash 499800 + 120 shares
    ]


# Testing whether phantom short exists or not.
# Set up FakePortfolio so that it contains 100 stocks of ticker AAA, then try to
# sell 100 in day 1, and 100 in day 2.
# It should emit a Warning.

def test_phantom_short_emits_warning(engine):
    sell_1 = quant.Order(
        ticker="AAA",
        action="sell",
        amount=100,
        date=pd.Timestamp("2020-01-04"),
        price=10.0,
    )
    sell_2 = quant.Order(
        ticker="AAA",
        action="sell",
        amount=100,
        date=pd.Timestamp("2020-01-05"),
        price=10.0,
    )
    portfolio = FakePortfolio([{"AAA": sell_1}, {"AAA": sell_2}])
    portfolio.stocks = {"AAA": 100}

    with pytest.raises(ValueError):
        engine.simulate(portfolio)

    # Only the first sell went through; the second was rejected.
    assert len(portfolio.t0_order) == 1
    # No phantom short: holdings never went negative, and the first sell
    # never even settled.
    assert portfolio.stocks["AAA"] == 0
    assert portfolio.t2_order == []


# Testing whether double spending exists or not.
# Set up FakePortfolio so that it could only buy 100 stocks of ticker AAA in day 1,
# then try to buy another 100 stocks of ticker AAA in day 2.
# The function should emit a Warning.

def test_double_spending_emits_warning(engine):
    buy_1 = quant.Order(
        ticker="AAA",
        action="buy",
        amount=100,
        date=pd.Timestamp("2020-01-04"),
        price=10.0,
    )
    buy_2 = quant.Order(
        ticker="AAA",
        action="buy",
        amount=100,
        date=pd.Timestamp("2020-01-05"),
        price=10.0,
    )
    portfolio = FakePortfolio([{"AAA": buy_1}, {"AAA": buy_2}])
    # Exactly enough cash for one 100-stock buy at 10.0.
    portfolio.current_cash = 1000

    with pytest.raises(ValueError):
        engine.simulate(portfolio)

    # Only the first buy went through; the second was rejected.
    assert len(portfolio.t0_order) == 1
    assert portfolio.current_cash == 0
    # The failed buy never settled, so no stocks were delivered.
    assert portfolio.stocks == {}
    assert portfolio.t2_order == []


# Testing if compute_order returning None results in a no-op.

def test_compute_order_none_is_noop(engine):
    portfolio = FakePortfolio([])
    engine.simulate(portfolio)

    assert portfolio.t0_order == []
    assert portfolio.t2_order == []
    assert engine.trade_log == []
    assert portfolio.stocks == {}
    assert portfolio.current_cash == 500000
    assert (portfolio.portfolio_value == 500000).all()
