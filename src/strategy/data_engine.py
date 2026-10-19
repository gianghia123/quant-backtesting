"""Data layer for the VN100 backtester: historical daily OHLCV and today's
opening prices, both from the DNSE OpenAPI."""
import os
import time
import json
from datetime import date, datetime, timedelta, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dnse import DNSEClient
from dotenv import load_dotenv

load_dotenv()

VN_TZ_NAME = "Asia/Ho_Chi_Minh"
VN_TZ = ZoneInfo(VN_TZ_NAME)
MARKET_DONE = dtime(15, 0)      # after this, today's daily candle counts as complete
PRICES_PATH = Path("prices.pkl")
DEFAULT_START = "2026-05-01"
PRICE_SCALE = 1.0               # DNSE sample shows 23.8 for ACB -> assumed already x1000 VND (verify)

UNIVERSE = [
    'ACB', 'BCM', 'BID', 'BVH', 'CTG',
    'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
    'MBB', 'MSN', 'MWG', 'PLX', 'POW',
    'SAB', 'SHB', 'SSB', 'SSI', 'STB',
    'TCB', 'TPB', 'VCB', 'VHM', 'VIB',
    'VIC', 'VJC', 'VNM', 'VPB', 'VRE'
]

OHLCV_COLS = ["adj_open", "adj_high", "adj_low", "adj_close", "volume"]
PRICE_COLS = ["adj_open", "adj_high", "adj_low", "adj_close"]

_client = None


# --------------------------------------------------------------------------
# DNSE plumbing
# --------------------------------------------------------------------------
def _get_client() -> DNSEClient:
    global _client
    if _client is None:
        _client = DNSEClient(
            api_key=os.environ["API_KEY"],
            api_secret=os.environ["API_SECRET"],
            base_url="https://openapi.dnse.com.vn",
            api_version=os.environ.get("DNSE_API_VERSION", "2026-07-23"),
        )
    return _client


def _to_ts(day, plus_days: int = 0) -> int:
    """'YYYY-MM-DD' (or date/Timestamp) -> Unix seconds at 00:00 Vietnam time."""
    d = datetime.strptime(str(day)[:10], "%Y-%m-%d").replace(tzinfo=VN_TZ)
    return int((d + timedelta(days=plus_days)).timestamp())

def _fetch_candles(client, symbol, resolution, start_ts, end_ts,
                   dry_run=False, max_pages=50):
    """Raw candles [(t, o, h, l, c, v), ...] for one symbol, following nextTime paging."""
    rows = []
    cursor = start_ts
    for _ in range(max_pages):
        status, body = client.get_ohlc(
            "STOCK",
            query={"symbol": symbol, "resolution": resolution,
                   "from": cursor, "to": end_ts},
            dry_run=dry_run,
        )
        if dry_run:
            return rows
        if status != 200:
            raise RuntimeError(f"HTTP {status}: {body}")

        # The SDK hands back the raw response text, so parse it here.
        if isinstance(body, (str, bytes)):
            try:
                body = json.loads(body)
            except ValueError:
                raise RuntimeError(f"Non-JSON response: {str(body)[:200]}")

        rows.extend(zip(body["t"], body["o"], body["h"], body["l"], body["c"], body["v"]))
        nxt = body.get("nextTime", 0)
        if not nxt or nxt <= cursor:
            break
        cursor = nxt
    return rows

def _to_frame(rows, price_scale: float) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["t", *OHLCV_COLS])
    df[PRICE_COLS] = df[PRICE_COLS] * price_scale
    ts = (pd.to_datetime(df.pop("t"), unit="s", utc=True)
            .dt.tz_convert(VN_TZ_NAME).dt.tz_localize(None).dt.normalize())
    df.index = pd.DatetimeIndex(ts, name="date")
    df = df.sort_index()
    return df[~df.index.duplicated(keep="last")][OHLCV_COLS]


# --------------------------------------------------------------------------
# Historical data
# --------------------------------------------------------------------------
def fetch_ohlcv(
    tickers: list[str],
    start: str,
    end: str | None = None,
    price_scale: float = PRICE_SCALE,
    retries: int = 3,
    pause: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """
    Fetch daily OHLCV per ticker from DNSE. Returns {ticker: DataFrame} indexed by
    date (tz-naive, Vietnam calendar day) with columns adj_open/high/low/close, volume.
    Exceptions are retried with backoff; an empty result (e.g. weekend) is not.
    """
    client = _get_client()
    end = end or datetime.now(VN_TZ).strftime("%Y-%m-%d")
    start_ts, end_ts = _to_ts(start), _to_ts(end, plus_days=1)  # end day inclusive

    data: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for ticker in tickers:
        rows = None
        for attempt in range(1, retries + 1):
            try:
                rows = _fetch_candles(client, ticker, "1D", start_ts, end_ts)
                break
            except Exception as e:
                print(f"[{ticker}] attempt {attempt}/{retries} failed: {e}")
                time.sleep(pause * attempt * 2)

        if rows is None:
            failed.append(ticker)
        elif rows:
            data[ticker] = _to_frame(rows, price_scale)
        time.sleep(pause)

    if failed:
        print(f"Failed to fetch: {failed}")

    # Unit sanity check: target unit is x1000 VND, and no VN100 stock trades
    # above ~1,000k VND, so a larger value means the units are not what we assumed.
    if data:
        max_close = max(df["adj_close"].max() for df in data.values())
        if max_close > 1000:
            print(f"WARNING: max close = {max_close:,.0f}; units look wrong "
                  "(expected x1000 VND). Check PRICE_SCALE.")
    return data


def save_prices(data: dict[str, pd.DataFrame], path: str | Path = PRICES_PATH) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.to_pickle(data, tmp)
    tmp.replace(path)  # atomic swap so a crash never leaves a half-written file
    print(f"Saved {len(data)} tickers to {path}")


def create_date_info(price_data: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    return price_data['ACB'].index


def get_historical_data(path: str | Path = PRICES_PATH, default_start: str = DEFAULT_START):
    """
    Load the cached prices and top each ticker up to the last completed trading
    day. Tickers missing from the cache are fetched from `default_start`.
    Returns (price_data: dict[str, DataFrame], price_index: DatetimeIndex).
    """
    path = Path(path)
    now = datetime.now(VN_TZ)
    # Don't store today's candle until the session is over (it would be partial).
    cutoff = now.date() if now.time() >= MARKET_DONE else now.date() - timedelta(days=1)

    prices = pd.read_pickle(path) if path.exists() else {}
    changed = False

    for ticker in UNIVERSE:
        old = prices.get(ticker)
        if old is None or old.empty:
            start = date.fromisoformat(default_start)
        else:
            start = (old.index[-1] + pd.Timedelta(days=1)).date()
        if start > cutoff:
            continue  # already up to date

        new = fetch_ohlcv([ticker], start=str(start), end=str(cutoff)).get(ticker)
        if new is None or new.empty:
            continue

        merged = new if old is None or old.empty else pd.concat([old, new])
        prices[ticker] = merged[~merged.index.duplicated(keep="last")].sort_index()
        changed = True

    if changed:
        save_prices(prices, path)
    return prices, create_date_info(prices)


# --------------------------------------------------------------------------
# Today's opens
# --------------------------------------------------------------------------
def get_today_opens(symbols=None, resolution="1D", client=None, pause=0.1,
                    price_scale: float = PRICE_SCALE, dry_run=False):
    """
    Fetch today's open price for each ticker (default: UNIVERSE).

    Returns {symbol: {"open": float} | None | {"error": str}}
      - None  -> no candle yet today (before the ATO match, or non-trading day)
      - error -> request failed for that ticker only
    """
    client = client or _get_client()
    now = datetime.now(VN_TZ)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_ts = int(day_start.timestamp())
    end_ts = int((day_start + timedelta(days=1)).timestamp())

    results = {}
    for symbol in dict.fromkeys(symbols or UNIVERSE):  # dedupe, keep order
        try:
            rows = _fetch_candles(client, symbol, resolution, start_ts, end_ts,
                                  dry_run=dry_run)
            today = [(r[0], r[1]) for r in rows
                     if datetime.fromtimestamp(r[0], VN_TZ).date() == now.date()]
            results[symbol] = min(today)[1] * price_scale if today else None
        except Exception as e:
            results[symbol] = {"error": str(e)}
        time.sleep(pause)  # gentle on rate limits
    return results


if __name__ == "__main__":
    prices, dates = get_historical_data()
    first = next(iter(prices))
    print(f"{len(prices)} tickers, {dates[0].date()} -> {dates[-1].date()}")
    print(prices[first].tail(3), "\n")
    print(get_today_opens(["ACB", "VNM"]))
