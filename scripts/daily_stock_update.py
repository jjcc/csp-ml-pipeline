"""
Daily stock price update utility.

Usage:
    python scripts/daily_stock_update.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from datetime import datetime
import os
from a00build_dataset_with_features import ensure_cache_dir
from service.utils import get_symbols_last_few_days, download_prices_batched
from service.data_prepare import COMMON_START_DATE, _load_cached_price_data, _save_cached_price_data



def preload_prices_with_cache_by_time(syms, out_dir, batch_size=30, cut_off_date=None, check_date = None):
    """
    From the raw CSP rows, determine unique symbols and date window,
    load from cache if available, batch-download missing ones, and save to cache.
    Returns dict: symbol -> DataFrame with OHLCV data
    """
    from service.data_prepare import _save_cached_price_data, _load_cached_price_data
    cache_dir = ensure_cache_dir(out_dir)
    # Determine symbols and window
    #syms = raw_df['baseSymbol'].dropna().astype(str).str.upper().unique().tolist()
    # Window: from min(tradeTime, expiry)-5 days to max(expiry)+1 day
    #tt = pd.to_datetime(raw_df.get('tradeTime', pd.NaT), errors="coerce")
    #ed = pd.to_datetime(raw_df.get('expirationDate', pd.NaT), errors="coerce")

    start_dt = check_date - pd.Timedelta(days=32) # download is 30 days
    # check week day of start_dt
    if start_dt.weekday() == 5:  # Saturday
        start_dt += pd.Timedelta(days=2)
    elif start_dt.weekday() == 6:  # Sunday
        start_dt += pd.Timedelta(days=1)
    end_dt = check_date
    if cut_off_date is  None:
        cut_off_date = pd.Timestamp.now()
    if end_dt > cut_off_date:
        end_dt = cut_off_date
        end_dt = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    # check week day of end_dt
    if end_dt.weekday() == 5:  # Saturday
        end_dt -= pd.Timedelta(days=1)
    elif end_dt.weekday() == 6:  # Sunday
        end_dt -= pd.Timedelta(days=2)
    # Load from cache or mark missing
    prices = {}
    missing = []
    for s in syms:
        price_df, ready = _load_cached_price_data(cache_dir, s, check_time=end_dt)
        if ready and (price_df.index.min() <= start_dt) :
            prices[s] = price_df
        else:
            missing.append(s)
    if missing:
        print(f"[INFO] Downloading {len(missing)} symbols in batches (size={batch_size})...")
        st = pd.to_datetime(COMMON_START_DATE)
        fetched = download_prices_batched(missing, st, end_dt, batch_size=batch_size, threads=True)
        for s, price_df in fetched.items():
            prices[s] = price_df
            _save_cached_price_data(cache_dir, s, price_df)
    return prices

def get_today_and_prevday(minus = 0):
    """
    get today and previous day. if previous_day is Saturday or Sunday, adjust accordingly.
    """
    today = datetime.now()
    if minus > 0:
        today -= pd.Timedelta(days=minus)
    previous_day = today - pd.Timedelta(days=1)
    if previous_day.weekday() == 5:  # Saturday
        previous_day -= pd.Timedelta(days=1)
    elif previous_day.weekday() == 6:  # Sunday
        previous_day -= pd.Timedelta(days=2)
    return previous_day, today

def amend_prices(cache_dir, price_df_today, to_update):
    for s in to_update:
        price_df, ready = _load_cached_price_data(cache_dir, s)
        price_df_today_s = price_df_today.get(s)
        if not ready:
            print(f"Warn: Price data missed for {s}")
        price_df = pd.concat([price_df, price_df_today_s])
        _save_cached_price_data(cache_dir, s, price_df)

def stock_price_update(test = False):
    previous_day, today = get_today_and_prevday()
    print(f"Previous day: {previous_day}, Today: {today}")
    out_dir = os.getenv("CACHE_DIR", "./output")

    folder = "option/put"
    end_date = today
    # TODO: check the following line
    #end_date = today -pd.Timedelta(days=1) if today.hour < 16 else today
    files, symbols = get_symbols_last_few_days(folder, end_date)
    # With the symbols, check all the cached prices, check the latest date. For those with previous date, put into a list and update with today's price
    cache_dir = ensure_cache_dir(out_dir)

    to_reload = []
    to_update = []
    for s in symbols:
        price_df, ready = _load_cached_price_data(cache_dir, s)
        latest_date = price_df.index.max() if ready else None
        # in case latest_date is today then continue
        if latest_date and latest_date >= today.replace(hour=0, minute=0, second=0, microsecond=0):
            continue
        if not ready:
            to_reload.append(s)
            continue
        if latest_date and latest_date == previous_day.replace(hour=0, minute=0, second=0, microsecond=0):
            to_update.append(s)
            continue
        else:
            to_reload.append(s)

    # Update with today's price
    if test:
        to_update = to_update[:5]
    if len(to_update)>0:
        price_df = download_prices_batched(to_update, previous_day, today, batch_size=100, threads=True)
        price_df_today = {k:v[-1:] for k,v in price_df.items()}
        # amend the cache
        amend_prices(cache_dir, price_df_today, to_update)
    if len(to_reload)>0:
        #start_date = previous_day - pd.Timedelta(days=30)
        # use common start date to avoid other training/testing data missing
        start_date = pd.to_datetime(COMMON_START_DATE)
        fetched = download_prices_batched(to_reload, start_date, today, batch_size=40, threads=True)
        # save the prices
        for s, price_df in fetched.items():
            _save_cached_price_data(cache_dir, s, price_df)
    with open("../logs/price_update.log","a") as f:
        f.write(f"### Price update done at {datetime.now()} with update {len(to_update)}, reload: {len(to_reload)}\n")
        f.write(f"To update: {to_update}\n")
        f.write(f"To reload: {to_reload}\n")

def main():
    stock_price_update()

if __name__ == "__main__":
    main()
