#!/usr/bin/env python3
import json
import os
import numpy as np
import pandas as pd
from service.stock_data_manager import GroupedStockUpdater
from service.utils import download_prices_batched
import numpy as np
from pathlib import Path
import yfinance as yf

from service.constants import COMMON_START_DATE, DEFAULT_BATCH_SIZE, DEFAULT_SLEEP_SECONDS, WEEKDAY_SATURDAY, WEEKDAY_SUNDAY

# Project root for absolute paths
PROJECT_ROOT = Path(__file__).parent.parent
MISSING_STOCKS_PATH = PROJECT_ROOT / "data" / "missing_stocks.json"


# moved from a00build_basic_dataset.py
def _load_cached_price_data(cache_dir, symbol, check_time = None):
    p = os.path.join(cache_dir, f"{symbol}.parquet")
    check_time = check_time.replace(hour=0, minute=0, second=0, microsecond=0) if check_time is not None else None
    if os.path.exists(p):
        try:
            df = pd.read_parquet(p)
            df.index = pd.to_datetime(df.index)
            if check_time is not None:
                # compare max date in df with check_time
                if df.index.max() >=  check_time:
                    return df, True
                else:
                    return df, False
            return df, True
        except Exception as e:
            print(f"[WARN] cache read failed for {symbol}: {e}")
    return None, False

def _save_cached_price_data(cache_dir, symbol, price_df: pd.DataFrame):
    try:
        price_df.to_parquet(os.path.join(cache_dir, f"{symbol}.parquet"))
    except Exception as e:
        print(f"[WARN] cache write failed for {symbol}: {e}")

def preload_prices_with_cache(syms,tt, ed, out_dir, batch_size=30, cut_off_date=None):
    """
    From the raw CSP rows, determine unique symbols and date window,
    load from cache if available, batch-download missing ones, and save to cache.
    Returns dict: symbol -> DataFrame with OHLCV data
    """
    cache_dir = os.path.join(out_dir, "price_cache")
    os.makedirs(cache_dir, exist_ok=True)

    with open(MISSING_STOCKS_PATH, "r") as f:
        missing_stocks = json.load(f)

    # Determine symbols and window
    # syms = raw_df['baseSymbol'].dropna().astype(str).str.upper().unique().tolist()
    # Window: from min(tradeTime, expiry)-5 days to max(expiry)+1 day
    #tt = pd.to_datetime(raw_df.get('tradeTime', pd.NaT), errors="coerce")
    #ed = pd.to_datetime(raw_df.get('expirationDate', pd.NaT), errors="coerce")
    start_dt = pd.to_datetime(min([d for d in pd.concat([tt, ed.dropna()]) if pd.notna(d)]) - pd.Timedelta(days=5))
    # check week day of start_dt
    if start_dt.weekday() == WEEKDAY_SATURDAY:  # Saturday
        start_dt += pd.Timedelta(days=2)
    elif start_dt.weekday() == WEEKDAY_SUNDAY:  # Sunday
        start_dt += pd.Timedelta(days=1)
    end_dt = pd.to_datetime(max([d for d in ed.dropna()]) + pd.Timedelta(days=1))
    if cut_off_date is  None:
        cut_off_date = pd.Timestamp.now()
        if cut_off_date.hour < 17:
            cut_off_date = cut_off_date - pd.Timedelta(days=1)
    if end_dt > cut_off_date:
        end_dt = cut_off_date
        end_dt = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    # check week day of end_dt
    if end_dt.weekday() == WEEKDAY_SATURDAY:  # Saturday
        end_dt -= pd.Timedelta(days=1)
    elif end_dt.weekday() == WEEKDAY_SUNDAY:  # Sunday
        end_dt -= pd.Timedelta(days=2)
    # Load from cache or mark missing
    prices = {}
    missing = []
    for s in syms:
        if '.' in s: # convert dot to hyphen for yfinance, like BRK.B to BRK-B
            s = s.replace('.', '-')
        if s in missing_stocks:
            print(f"[INFO] Skipping known stock missing in yfinance: {s}")
            continue
        price_df, _ = _load_cached_price_data(cache_dir, s)
        if price_df is not None and (price_df.index.min() <= start_dt) and (price_df.index.max() >= end_dt):
            # convert the '-' back to '.' for BRK.B
            if '-' in s:
                s = s.replace('-', '.')
            prices[s] = price_df
        else:
            missing.append(s)
    if missing:
        print(f"[INFO] Downloading {len(missing)} symbols in batches (size={batch_size})...")
        log_file = "data/price_cache_dates.csv"

        updater = GroupedStockUpdater(data_dir=cache_dir, log_file=log_file)
        _,_,fetched = updater.update_batch(missing)
        prices.update(fetched)
        #fetched_start = start_dt - pd.Timedelta(days=10)
        #st = pd.to_datetime(COMMON_START_DATE)
        #fetched_end = pd.Timestamp.now()
        #fetched = download_prices_batched(missing, st, fetched_end, batch_size=batch_size, threads=True)
        #for s, price_df in fetched.items():
        #    prices[s] = price_df
        #    _save_cached_price_data(cache_dir, s, price_df)
    return prices

def lookup_close_on_or_before(price_df: pd.DataFrame, target_dt: pd.Timestamp, max_days_back: int = 3) -> float:
    """
    Lookup close price on or before target date, but only if it's within max_days_back.
    This prevents using stale prices from many days ago for future expiration dates.

    Args:
        price_df: DataFrame with Close prices indexed by date
        target_dt: Target expiration date
        max_days_back: Maximum days back to look for a valid price (default: 3)

    Returns:
        Close price if found within max_days_back, otherwise np.nan
    """
    if price_df is None or price_df.empty or pd.isna(target_dt) or 'Close' not in price_df.columns:
        return np.nan

    target_dt = pd.to_datetime(target_dt)

    # Only look for prices within max_days_back of the target date
    min_date = target_dt - pd.Timedelta(days=max_days_back)
    sub = price_df[(price_df.index >= min_date) & (price_df.index <= target_dt)]

    if len(sub) == 0:
        return np.nan

    # Return the closest price to expiration date within the window
    return float(sub['Close'].iloc[-1])

def derive_capital(df, policy="strike100", constant_capital=10_000.0):
    strike = pd.to_numeric(df["strike"], errors="coerce")
    ul = pd.to_numeric(df.get("underlyingLastPrice"), errors="coerce")
    # per-share option price at entry (fallback to bidPrice)
    opt_px = pd.to_numeric(df.get("bid", df.get("bidPrice")), errors="coerce")
    entry_credit = pd.to_numeric(df["entry_credit"], errors="coerce")  # already *100

    if policy == "constant":
        df["capital"] = float(constant_capital)
        cap = df["capital"]
    elif policy == "strike100":
        cap = 100.0 * strike
    elif policy == "credit_adjusted":
        cap = 100.0 * strike - entry_credit
    elif policy == "regt_light":
        # Rough Reg-T proxy for short puts: per-share margin = option_px + max(0.20*UL - OTM, 0.10*UL)
        otm = np.maximum(ul - strike, 0.0)
        per_share_margin = opt_px + np.maximum(0.20*ul - otm, 0.10*ul)
        cap = 100.0 * per_share_margin
    else:
        cap = 100.0 * strike

    cap = pd.to_numeric(cap, errors="coerce").replace([np.inf,-np.inf], np.nan)
    # sensible floor to avoid divide-by-zero
    return pd.Series(cap).fillna(100.0 * strike).clip(lower=50.0)

# End moved from a00build_basic_dataset.py

# moved from a02merge_macro_features.py
def _coerce_dt(s):
    return pd.to_datetime(s, errors="coerce")

def _load_vix(vix_csv, start_date, end_date,force_download=False):
    # Return a Series indexed by date (daily), name="VIX"
    if vix_csv and Path(vix_csv).exists():
        vdf = pd.read_csv(vix_csv)
        date_col = "Date" if "Date" in vdf.columns else "date"
        close_col = "VIX" if "VIX" in vdf.columns else "Close"
        vdf[date_col] = pd.to_datetime(vdf[date_col], errors="coerce")
        max_date = vdf[date_col].max()
        if pd.isna(max_date) or max_date >= (pd.to_datetime(end_date) - pd.Timedelta(days=1)) and not force_download:
            vdf = vdf.dropna(subset=[date_col]).set_index(date_col).sort_index()
            return vdf.loc[start_date:end_date, close_col].rename("VIX").astype(float)
    #if use_yf:
    try:
        if force_download:
            end_date = pd.Timestamp.now().date()
        df = yf.download("^VIX", start=start_date, end=end_date)
        # Handle MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            # Get single-index DataFrame by selecting the ^VIX ticker level
            vix_df = df.xs("^VIX", axis=1, level=1)  # Extract ^VIX columns, drop MultiIndex
            s = vix_df["Close"].rename("VIX")
        else:
            s = df["Close"].rename("VIX")
        s.index = pd.to_datetime(s.index)
        s.to_csv(vix_csv)
        return s
    except Exception:
        pass
    return pd.Series(dtype=float, name="VIX")

def _load_symbol_prices(symbol, px_dir, start_date, end_date, use_yf=False):
    # Return a Series indexed by date (business days ok), name="Close"
    if px_dir:
        #f = Path(px_dir) / f"{symbol}.csv"
        f = Path(px_dir) / f"{symbol}.parquet"
        if f.exists():
            #df = pd.read_csv(f)
            df = pd.read_parquet(f)
            if df.columns.nlevels > 1:
                # MultiIndex columns from yfinance
                df = df[symbol] # Extract symbol columns, drop MultiIndex
            # Check if index is already datetime, otherwise look for Date/date columns
            if pd.api.types.is_datetime64_any_dtype(df.index):
                df = df.sort_index()
                if symbol == "WOLF": # special case
                    # check if the end_date is less than "2015-09-27"
                    if end_date < pd.to_datetime("2025-09-27"):
                        # reload df from another file
                        df = pd.read_parquet("data/wolf/WOLF.parquet") # this is an old price which was overridden in yfinance
                        df.sort_index()
                    close_col = "Close" if "Close" in df.columns else "close"
                    return df.loc[start_date:, close_col].rename("Close").astype(float)
                # check if min and max date cover the range
                #if  start_date >= df.index.min() and end_date - pd.Timedelta(days=1) <= df.index.max():
                if len(df) > 0 :
                    close_col = "Close" if "Close" in df.columns else "close"
                    if symbol == "NFLX":
                        # special case for NFLX with duplicated index
                        close_col = "Adj Close" if "Adj Close" in df.columns else close_col
                    return df.loc[start_date:end_date, close_col].rename("Close").astype(float)
                else:
                    # update the price file by downloading from yf
                    use_yf = True
                    return None
            else:
                date_col = "Date" if "Date" in df.columns else "date"
                close_col = "Close" if "Close" in df.columns else "close"
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
                return df.loc[start_date:end_date, close_col].rename("Close").astype(float)
    if use_yf and symbol:
        raise NotImplementedError("Yahoo Finance integration not implemented")
        try:
            import yfinance as yf
            s = yf.download(symbol, start=start_date, end=end_date)
            s = s["Close"].rename("Close")
            s.index = pd.to_datetime(s.index)
            return s
        except Exception:
            pass
    return pd.Series(dtype=float, name="Close")

def _per_symbol_feature_frame(s_px: pd.Series, start_date, max_trade_date) -> pd.DataFrame:
    """
    Build per-symbol daily features with a row for every business day up to max_trade_date,
    but never fabricate today's (T) close if it doesn't exist. No leakage:
      prev_close(T)      = Close(T-1)
      ret_2d(T), ret_5d(T) use returns up to T-1
    """
    import numpy as np
    import pandas as pd

    if s_px is None or s_px.empty:
        # still return an empty frame on the requested calendar so merge works
        cal = pd.bdate_range(start=start_date, end=max_trade_date)
        return pd.DataFrame(index=cal)

    s_px = s_px.sort_index()
    last_px_date = s_px.index.max()

    # Calendar up to trade-date T (even if T's close is missing)
    cal = pd.bdate_range(start=start_date, end=max_trade_date)

    # Reindex to calendar; ffill only up to the last REAL close
    try:
        daily = s_px.reindex(cal)
    except Exception as ex:
        print(f"Reindexing error for symbol prices: {ex}")
        # return empty dataframe
        return pd.DataFrame(index=cal)
    # Forward-fill holiday gaps with the most recent PAST close (no leakage).
    # bfill would use the next future close for holidays, which is future leakage.
    daily_ff = daily.ffill()
    # DO NOT fill beyond the last real price date (keeps future dates as NaN)
    daily_ff.loc[daily_ff.index > last_px_date] = np.nan

    # 1-day returns on ffilled series; today's ret is NaN if today's Close is NaN
    ret1d = daily_ff.pct_change()
    # strict lag so all rolling windows end at T-1
    ret1d_lag = ret1d.shift(1)

    out = pd.DataFrame(index=cal)
    out["prev_close"]   = daily_ff.shift(1)                     # Close(T-1)
    # round to 4 decimal places
    out["prev_close"] = out["prev_close"].round(4)
    out["ret_2d"]       = ret1d_lag.rolling(2, min_periods=2).sum()
    out["ret_5d"]       = ret1d_lag.rolling(5, min_periods=5).sum()
    vol20               = ret1d_lag.rolling(20, min_periods=5).std()
    out["ret_2d_norm"]  = out["ret_2d"] / vol20.replace(0, np.nan)
    out["ret_5d_norm"]  = out["ret_5d"] / vol20.replace(0, np.nan)
    return out

def per_symbol_price_feat(PX_BASE_DIR, df, vix_df, need_symbol, buffer_days=60, loaded_prices=None):
    if not need_symbol:
        raise SystemExit("Expected baseSymbol column for per-symbol price features.")

    # Compute DTE features on the row level (calendar days, clipped to ≥1)
    d = df.copy()

    d["log1p_DTE"] = np.log1p(d["daysToExpiration"].astype(float))

    # Will collect per-symbol PX frames and merge back row-wise
    feats = []
    d["trade_date"] = pd.to_datetime(d["trade_date"], errors="coerce").dt.floor("D")
    for sym in d["baseSymbol"].dropna().unique():
        s_mask = d["baseSymbol"] == sym
        start = d.loc[s_mask, "trade_date"].min() - pd.Timedelta(days=buffer_days)
        # IMPORTANT: end at the **max trade_date** (T), not +1BD in real time
        stop  = d.loc[s_mask, "trade_date"].max()
        s_px = loaded_prices.get(sym) if loaded_prices is not None else None
        # remove duplicated index
        s_px = s_px[~s_px.index.duplicated(keep="first")] if s_px is not None else None
        f = _per_symbol_feature_frame(s_px, start, stop)
        if f.empty:
            f = pd.DataFrame(index=pd.date_range(start, stop, freq="B"))

        f = f.reset_index().rename(columns={"index": "trade_date"})
        f["baseSymbol"] = sym
        feats.append(f)

    px_feat = (pd.concat(feats, ignore_index=True)
               if feats else pd.DataFrame(columns=["trade_date", "baseSymbol"]))

    # Merge VIX by date (then ffill gaps)
    d = d.merge(vix_df, on="trade_date", how="left")
    d["VIX"] = d["VIX"].ffill()

    # Merge per-date, per-symbol PX features — gives different values per trade date
    d = d.merge(px_feat, on=["trade_date","baseSymbol"], how="left")
    return d

def add_macro_features(df, vix_df_or_csv_path, px_base_dir):
    """
    Shared function to add macro features to a dataframe.
    Extracted from the main() function to be reusable by other scripts.

    Args:
        df: Input dataframe with tradeTime column
        vix_df_or_csv_path: Either a VIX DataFrame with columns [trade_date, VIX],
                           or a string path to VIX CSV file
        px_base_dir: Directory containing price data (PX_BASE_DIR environment variable)

    Returns:
        DataFrame with macro features added
    Change Log:
        - 2025-09-28: Load symbol prices first and check if avaiable. If not, download from yfinance and save to px_base_dir.
    """
    # Parse dates and derive trade_date (calendar day)
    df = df.copy()
    df["tradeTime"] = _coerce_dt(df["tradeTime"])
    df["expirationDate"] = _coerce_dt(df["expirationDate"])
    df["trade_date"] = df["tradeTime"].dt.floor("D")

    start_date = df["trade_date"].min() # - pd.Timedelta(days=60)
    end_date   = df["trade_date"].max() + pd.Timedelta(days=1)

    # VIX (global) - handle both DataFrame and CSV path
    if isinstance(vix_df_or_csv_path, pd.DataFrame):
        # Pre-built VIX DataFrame (from task_score_tail_winner.py)
        vix_df = vix_df_or_csv_path
    else:
        # CSV path (from a02merge_macro_features.py)
        vix = _load_vix(vix_df_or_csv_path, start_date, end_date)
        vix_df = pd.DataFrame({"trade_date": vix.index, "VIX": vix.values})

    # check all symbols for date range
    symbol_list = df['baseSymbol'].unique()
    to_reload = []
    loaded = {}
    st = start_date - pd.Timedelta(days=24) # add a time buffer
    ed = pd.Timestamp.now()
    for sym in symbol_list:
        s_px = _load_symbol_prices(sym, px_base_dir, st, end_date)
        if s_px is not None and not s_px.empty:
            s_px = s_px.sort_index()
            if  st < s_px.index.min() or (end_date - pd.Timedelta(days=1)) > s_px.index.max():
                to_reload.append(sym)
            else:
                loaded[sym] = s_px
        else:
            to_reload.append(sym)
    if len(to_reload):
        print(f"[INFO] Reloading {len(to_reload)} symbols from {px_base_dir} due to date range...")

        #for sym in to_reload:
        # batch reload prices
        st = pd.to_datetime(COMMON_START_DATE)
        fetched = download_prices_batched(to_reload, st, pd.Timestamp.now(), batch_size=30, threads=True)
        for s, price_df in fetched.items():
            _save_cached_price_data(px_base_dir, s, price_df)
            close_price = price_df["Close"]
            loaded[s] = close_price

    # Per-symbol price features
    need_symbol = "baseSymbol" in df.columns
    d = per_symbol_price_feat(px_base_dir, df, vix_df, need_symbol, buffer_days=28, loaded_prices=loaded)

    # Gap between previous close and current underlying price
    if "underlyingLastPrice" in d.columns:
        d["prev_close_minus_ul"] = d["prev_close"] - d["underlyingLastPrice"]
        d["prev_close_minus_ul_pct"] = (d["prev_close"] - d["underlyingLastPrice"]) / d["underlyingLastPrice"].replace(0, np.nan)
    else:
        d["prev_close_minus_ul"] = np.nan
        d["prev_close_minus_ul_pct"] = np.nan

    return d
# End moved from a02merge_macro_features.py
