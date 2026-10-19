import time
from datetime import date
from pathlib import Path

import pandas as pd
import quant_agent as qa

OHLCV_COLS = ["adj_open", "adj_high", "adj_low", "adj_close", "volume"]
PRICE_COLS = ["adj_open", "adj_high", "adj_low", "adj_close"]

RENAME = {
    "open": "adj_open",
    "high": "adj_high",
    "low": "adj_low",
    "close": "adj_close",
}


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    df = df.rename(columns=RENAME)

    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ("time", "date", "datetime"):
            if col in df.columns:
                df = df.set_index(col)
                break

    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "date"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    missing = [c for c in OHLCV_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}; got {list(df.columns)}")
    return df[OHLCV_COLS]  # also drops the 'ticker' column

def fetch_ohlcv(
    tickers: list[str],
    start: str,
    end: str | None = None,
    source: str | None = None,   # None -> library default (DNSE); "VCI" for long history
    price_scale: float = 1 / 1000,  # VND -> x1000 VND. Use 1.0 to keep raw VND.
    retries: int = 3,
    pause: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """
    Fetch daily OHLCV per ticker via quant_agent.

    Prices are requested in VND (beautify=True) and then multiplied by
    `price_scale`, so the default gives x1000 VND. Volume is not scaled.
    """
    end = end or date.today().strftime("%Y-%m-%d")
    kwargs = dict(
        start_date=start,
        end_date=end,
        resolution="1D",
        type="stock",
        beautify=True,   # documented to return prices in VND
        decor=True,
    )
    if source:
        kwargs["source"] = source

    data: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for ticker in tickers:
        df = None
        for attempt in range(1, retries + 1):
            try:
                out = qa.stock_historical_data(symbol=ticker, **kwargs)
                if out is not None and len(out) > 0:
                    df = _normalise(out)
                    break
                print(f"[{ticker}] attempt {attempt}/{retries}: empty result")
            except Exception as e:
                print(f"[{ticker}] attempt {attempt}/{retries} failed: {e}")
            time.sleep(pause * attempt * 2)

        if df is None or df.empty:
            failed.append(ticker)
        else:
            df[PRICE_COLS] = df[PRICE_COLS] * price_scale
            data[ticker] = df
        time.sleep(pause)

    if failed:
        print(f"Failed to fetch: {failed}")

    # Unit sanity check (only meaningful for the default x1000 VND scale):
    # no VN100 stock trades above ~1,000k VND, so a larger value means
    # the units are not what we assumed.
    if price_scale == 1 / 1000 and data:
        max_close = max(df["adj_close"].max() for df in data.values())
        if max_close > 1000:
            print(f"WARNING: max close = {max_close:,.0f}; units look wrong "
                  "(expected x1000 VND). Check beautify/price_scale.")
    return data


def save_prices(data: dict[str, pd.DataFrame], path: str | Path = "prices.pkl") -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.to_pickle(data, tmp)
    tmp.replace(path)  # atomic swap so a crash never leaves a half-written file
    print(f"Saved {len(data)} tickers to {path}")


if __name__ == "__main__":
    universe = [
        'ACB', 'BCM', 'BID', 'BVH', 'CTG',
        'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
        'MBB', 'MSN', 'MWG', 'PLX', 'POW',
        'SAB', 'SHB', 'SSB', 'SSI', 'STB',
        'TCB', 'TPB', 'VCB', 'VHM', 'VIB',
        'VIC', 'VJC', 'VNM', 'VPB', 'VRE'
    ]
    prices = fetch_ohlcv(universe, start="2026-05-01")
    save_prices(prices, "prices.pkl")

    # Later / in your backtester:
    # prices = load_prices("prices.pkl")

    for t, df in prices.items():
        print(t, df.shape)
        print(df.tail(3), "\n")
