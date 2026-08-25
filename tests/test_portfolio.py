import pickle

import pytest
import pandas as pd

import quant
from quant.portfolio import Portfolio


@pytest.fixture
def simple_portfolio(monkeypatch):
    def date_index(*args, **kwargs):
        return pd.date_range("2020-01-04", "2020-01-07")
    monkeypatch.setattr(pickle, "load", date_index)
    p = quant.Portfolio(
        "/home/supperman/.zshrc",
        init_cash=1000,
        slippage=0.0,
        commission_rate=0.0,
        tax_rate=0.0
    )
    return p

@pytest.fixture
def price_data(simple_portfolio):
    simple_portfolio.stocks = {"AAA": 100, "BBB": 200}
    price_data = [
        {"AAA": 5, "BBB": 7},
        {"AAA": 4, "BBB": 8},
        {"AAA": 6, "BBB": 5},
        {"AAA": 5, "BBB": 5}
    ]
    return simple_portfolio, price_data

@pytest.fixture
def buy_sell_data(simple_portfolio):
    simple_portfolio.stocks = {"AAA": 100}
    sell_order = quant.Order(
        action="sell",
        ticker="AAA",
        amount=50,
        price=10,
        date=pd.Timestamp("2020-01-04")
    )
    buy_order = quant.Order(
        action='buy',
        ticker='AAA',
        amount=50,
        price=10,
        date=pd.Timestamp("2020-01-04")
    )
    return simple_portfolio, sell_order, buy_order

def test_buy_simple_t0(buy_sell_data):
    p: Portfolio = buy_sell_data[0]
    buy_order = buy_sell_data[2]
    p.execute_order_t0(buy_order)
    assert p.current_cash == 1000 - 50 * 10
    assert p.stocks == {"AAA": 100}

def test_buy_simple_t2(buy_sell_data):
    p: Portfolio = buy_sell_data[0]
    buy_order = buy_sell_data[2]
    p.execute_order_t0(buy_order)
    p.execute_order_t2(buy_order)
    assert p.current_cash == 1000 - 50 * 10
    assert p.stocks == {"AAA": 100 + 50}

def test_sell_simple_t0(buy_sell_data):
    p: Portfolio = buy_sell_data[0]
    sell_order = buy_sell_data[1]
    p.execute_order_t0(sell_order)
    assert p.current_cash == 1000
    assert p.stocks == {"AAA": 100 - 50}

def test_sell_simple_t2(buy_sell_data):
    p: Portfolio = buy_sell_data[0]
    sell_order = buy_sell_data[1]
    p.execute_order_t0(sell_order)
    p.execute_order_t2(sell_order)
    assert p.current_cash == 1000 + 50 * 10
    assert p.stocks == {"AAA": 100 - 50}

def test_update_value_day_1(price_data):
    p: Portfolio = price_data[0]
    price_data_day_1 = price_data[1][0]
    date = p.date_index[0]
    p.update_value(date, price_data_day_1)
    assert p.portfolio_value.loc[date] == 1000 + (100 * 5) + (200 * 7)

def test_update_value_all(price_data):
    p: Portfolio = price_data[0]
    _price_data = price_data[1]
    dates = p.date_index
    targets = [2900, 3000, 2600, 2500]
    for indx, date in enumerate(dates):
        p.update_value(date, _price_data[indx])
        assert p.portfolio_value.iloc[indx] == targets[indx]
