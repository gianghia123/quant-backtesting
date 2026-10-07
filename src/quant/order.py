from dataclasses import dataclass, field

import pandas as pd

@dataclass
class Order:
    date: pd.Timestamp = field(init=False)
    ticker: str
    action: str
    amount: int
    price: float = field(init=False)

    def __post_init__(self):
        if self.action not in ['buy', 'sell']:
            raise ValueError(f'Action should be "buy" or "sell", not {self.action}')


