#!/usr/bin/env python3
"""
a04_label_data.py — Step 4: Label option trades with win/loss outcomes.

Fetches expiry-date closing prices via yfinance (with caching), computes PnL
and return metrics, and writes labeled CSV files ready for model training.

Two labeling modes (controlled by `merge_mode` flag at bottom of file):
  - Single-dataset mode (default): labels each individual enriched dataset
  - Merge mode: labels combined (merged) datasets in output/data_merged/

Symbol exclusions are loaded from data files:
  - data/missing_stocks.json  — symbols with no price data (auto-managed)
  - data/exclude_stocks.json  — manually curated exclusions

Usage:
    python pipeline/a04_label_data.py

Configuration:
    Reads from config.yaml and .env.  The cutoff date for each dataset tag is
    derived automatically from common_configs in config.yaml.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Ensure project root is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import exchange_calendars as xcals
    nyse = xcals.get_calendar("XNYS")
except Exception:
    nyse = None  # fall back to business-day heuristic

from service.data_prepare import derive_capital, preload_prices_with_cache
from service.env_config import get_derived_file, getenv

# ---------------------------------------------------------------------------
# Paths (absolute, robust to CWD changes)
# ---------------------------------------------------------------------------
PROJECT_ROOT        = Path(__file__).resolve().parent.parent
MISSING_STOCKS_PATH = PROJECT_ROOT / "data" / "missing_stocks.json"
EXCLUDE_STOCKS_PATH = PROJECT_ROOT / "data" / "exclude_stocks.json"


# ---------------------------------------------------------------------------
# Price / calendar helpers
# ---------------------------------------------------------------------------

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def resolve_last_trading_session(expiry_ts: pd.Timestamp) -> pd.Timestamp:
    """Return the last trading session on or before expiry_ts."""
    d = pd.Timestamp(expiry_ts).tz_localize(None).normalize()
    if nyse is not None:
        if nyse.is_session(d):
            sess = nyse.date_to_session(d, direction="none")
        else:
            sess = nyse.date_to_session(d, direction="previous")
        return pd.Timestamp(sess).normalize()
    # Fallback: step back weekends / non-Fridays
    while d.weekday() > 4:
        d -= pd.tseries.offsets.BDay(1)
    if d.weekday() != 4:
        d -= pd.tseries.offsets.BDay(1)
    return d.normalize()


def get_close_on_session(price_df, session_date, use_unadjusted: bool = True):
    if price_df is None or len(price_df) == 0:
        return np.nan
    if "date" in price_df.columns:
        idx = pd.to_datetime(price_df["date"]).dt.normalize()
        price_df = price_df.assign(_idx=idx).set_index("_idx")
    col = "Adj Close" if not use_unadjusted and "Adj Close" in price_df.columns else "Close"
    return float(price_df[col].get(session_date.normalize(), np.nan))


# ---------------------------------------------------------------------------
# Core labeling logic
# ---------------------------------------------------------------------------

def build_labeled_dataset(raw: pd.DataFrame, preload_closes: dict = None) -> pd.DataFrame:
    """
    Label raw trades with expiry outcomes (win/loss, PnL, returns).

    Required columns: baseSymbol, expirationDate, strike, tradeTime, underlyingLastPrice
    Optional columns are filled with NaN if absent.
    """
    df = raw.copy()

    required = ["baseSymbol", "expirationDate", "strike", "tradeTime", "underlyingLastPrice"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    optional = [
        "delta", "moneyness", "impliedVolatilityRank1y", "potentialReturn",
        "potentialReturnAnnual", "breakEvenProbability", "percentToBreakEvenBid",
        "openInterest", "volume", "__source_file",
    ]
    for c in optional:
        if c not in df.columns:
            df[c] = np.nan

    df["tradeTime"]      = pd.to_datetime(df["tradeTime"],      errors="coerce")
    df["expirationDate"] = pd.to_datetime(df["expirationDate"], errors="coerce")

    # ------------------------------------------------------------------
    # Look up expiry close price from pre-loaded cache
    # ------------------------------------------------------------------
    def expiry_close_from_cache(r):
        if preload_closes is None:
            return np.nan
        expiry_date = pd.to_datetime(r["expirationDate"], errors="coerce")
        if pd.isna(expiry_date) or expiry_date > pd.Timestamp.now():
            return np.nan
        session   = resolve_last_trading_session(expiry_date)
        sym       = str(r["baseSymbol"]).upper()
        price_df  = preload_closes.get(sym)
        if price_df is not None and price_df.columns.nlevels > 1:
            price_df = price_df[sym]
        return get_close_on_session(price_df, session, use_unadjusted=True)

    df["expiry_close"] = df.apply(expiry_close_from_cache, axis=1)

    # ------------------------------------------------------------------
    # Entry credit (use bid price)
    # ------------------------------------------------------------------
    def entry_credit(r, take_from_mid_pct: float = 0.35, min_abs: float = 0.01):
        bid_price = safe_float(r.get("bidPrice"))
        if not np.isfinite(bid_price):
            return np.nan
        fill = bid_price - max(min_abs, take_from_mid_pct * 0.0)
        return max(0.0, fill) * 100.0

    df["entry_credit"] = df.apply(entry_credit, axis=1)

    # ------------------------------------------------------------------
    # Exit intrinsic value at expiry (put option)
    # ------------------------------------------------------------------
    def exit_intrinsic(r):
        strike      = safe_float(r["strike"])
        expiry_close = safe_float(r["expiry_close"])
        if not np.isfinite(strike) or not np.isfinite(expiry_close):
            return np.nan
        return max(0.0, strike - expiry_close) * 100.0

    df["exit_intrinsic"] = df.apply(exit_intrinsic, axis=1)

    # ------------------------------------------------------------------
    # PnL and return metrics
    # ------------------------------------------------------------------
    df["capital"]    = derive_capital(df)
    df["total_pnl"]  = df["entry_credit"] - df["exit_intrinsic"]
    df["return_pct"] = np.where(df["capital"] > 0,
                                df["total_pnl"] / df["capital"] * 100.0,
                                np.nan)
    df["won"] = df["total_pnl"] > 0

    return df


# ---------------------------------------------------------------------------
# Exclusion helpers
# ---------------------------------------------------------------------------

def load_exclude_symbols() -> set:
    """Return combined set of symbols from missing_stocks.json and exclude_stocks.json."""
    excluded = set()

    if MISSING_STOCKS_PATH.exists():
        with open(MISSING_STOCKS_PATH) as f:
            excluded.update(json.load(f))

    if EXCLUDE_STOCKS_PATH.exists():
        with open(EXCLUDE_STOCKS_PATH) as f:
            data = json.load(f)
            # Support both list and dict formats
            excluded.update(data if isinstance(data, list) else data.keys())

    return excluded


# ---------------------------------------------------------------------------
# Dataset tag + cutoff helpers
# ---------------------------------------------------------------------------

def get_cutoff_dates() -> Dict[str, str]:
    """Return {dataset_tag: cutoff_date} from common_configs in config.yaml."""
    from service.env_config import config

    cutoff_by_tag: Dict[str, str] = {}
    for _key, cfg in config.get_common_configs_raw().items():
        if not isinstance(cfg, dict):
            continue
        basic_csv = cfg.get("data_basic_csv", "")
        stem      = basic_csv.replace(".csv", "")
        parts     = stem.split("_")
        try:
            tag = parts[parts.index("raw") + 1]
        except (ValueError, IndexError):
            continue
        cutoff = cfg.get("cutoff_date")
        if cutoff:
            cutoff_by_tag[tag] = cutoff

    return cutoff_by_tag


def extract_tag_from_filename(fname: str, merged: bool = False) -> str:
    """Derive dataset tag from an enriched/merged filename.

    Non-merged filenames look like: trades_with_gex_macro_<tag>_<date>.csv
    Merged filenames look like:     merged_with_gex_macro_<combo>.csv
                                    e.g. merged_with_gex_macro_origabcde.csv  → 'e'
    """
    stem  = Path(fname).stem
    parts = stem.split("_")

    if not merged:
        # e.g. trades_with_gex_macro_f_1027  → tag at index after 'macro'
        try:
            idx = parts.index("macro")
            return parts[idx + 1]
        except (ValueError, IndexError):
            return ""
    else:
        # last segment is the profile combo, last char is the tag (or 'orig')
        combo = parts[-1]
        return "orig" if combo == "orig" else combo[-1]


# ---------------------------------------------------------------------------
# CSV-level labeling
# ---------------------------------------------------------------------------

def label_csv_file(df: pd.DataFrame, output_csv: str, cut_off_date) -> None:
    """Label *df* and write to output/data_labeled/<output_csv>."""
    df["expirationDate"] = pd.to_datetime(df["expirationDate"], errors="coerce")
    cut_off_date         = pd.to_datetime(cut_off_date).normalize()

    # Filter: only trades that have expired by cut_off_date
    before = len(df)
    df = df[df["expirationDate"].notna() & (df["expirationDate"] <= cut_off_date)].copy()
    if len(df) != before:
        print(f"  Cutoff {cut_off_date}: removed {before - len(df)} future trades, {len(df)} remain.")

    # Filter: DTE <= 14
    before2 = len(df)
    df = df[df["daysToExpiration"] <= 14]
    if len(df) != before2:
        print(f"  DTE > 14 filter: removed {before2 - len(df)}, {len(df)} remain.")

    if df.empty:
        print(f"  [WARN] No rows to label after filters — skipping {output_csv}.")
        return

    # Preload prices
    cache_dir   = getenv("COMMON_OUTPUT_DIR", "./output")
    batch_size  = int(getenv("DATA_BATCH_SIZE", "30"))
    syms = df["baseSymbol"].dropna().astype(str).str.upper().unique().tolist()
    tt   = pd.to_datetime(df.get("tradeTime",      pd.NaT), errors="coerce")
    ed   = pd.to_datetime(df.get("expirationDate", pd.NaT), errors="coerce")

    closes = preload_prices_with_cache(syms, tt, ed, cache_dir,
                                       batch_size=batch_size,
                                       cut_off_date=cut_off_date)

    labeled = build_labeled_dataset(df, preload_closes=closes)
    labeled  = labeled[~labeled["won"].isna()].copy()

    print(f"  label_coverage={len(labeled)/max(len(df),1):.2%}  "
          f"win_rate={labeled['won'].mean():.2%}")

    out_dir = os.path.join(getenv("COMMON_OUTPUT_DIR", "./output"), "data_labeled")
    os.makedirs(out_dir, exist_ok=True)
    labeled.to_csv(os.path.join(out_dir, output_csv), index=False)
    print(f"  → {out_dir}/{output_csv}")


# ---------------------------------------------------------------------------
# Multi-dataset labeling: single (non-merged)
# ---------------------------------------------------------------------------

def label_multiple_single_datasets() -> None:
    """Label all individual enriched datasets found in output/data_prep/."""
    input_dir  = os.path.join(getenv("COMMON_OUTPUT_DIR", "output"), "data_prep")
    cutoff_map = get_cutoff_dates()
    exclude    = load_exclude_symbols()

    files = sorted(f for f in os.listdir(input_dir)
                   if f.startswith("trades_with_gex") and f.endswith(".csv"))

    if not files:
        print(f"[WARN] No enriched trade files found in {input_dir}")
        return

    for fname in files:
        fpath = os.path.join(input_dir, fname)
        print(f"\nProcessing: {fname}")
        df = pd.read_csv(fpath, index_col="row_id")

        # Apply exclusions from data files (no hard-coded lists)
        before = len(df)
        df = df[~df["baseSymbol"].isin(exclude)].copy()
        if len(df) != before:
            print(f"  Excluded {before - len(df)} rows from known bad symbols.")

        tag         = extract_tag_from_filename(fname, merged=False)
        cutoff_date = cutoff_map.get(tag)
        if cutoff_date is None:
            print(f"  [WARN] No cutoff date for tag '{tag}' — skipping.")
            continue

        output_csv = f"labeled_{fname}_filtered.csv"
        label_csv_file(df, output_csv, cutoff_date)


# ---------------------------------------------------------------------------
# Multi-dataset labeling: merged
# ---------------------------------------------------------------------------

def label_merged_datasets() -> None:
    """Label all merged datasets found in output/data_merged/."""
    input_dir  = os.path.join(getenv("COMMON_OUTPUT_DIR", "output"), "data_merged")
    cutoff_map = get_cutoff_dates()

    if not os.path.isdir(input_dir):
        print(f"[WARN] Merged input directory not found: {input_dir}")
        return

    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".csv"))
    for fname in files:
        fpath = os.path.join(input_dir, fname)
        print(f"\nProcessing merged: {fname}")
        df = pd.read_csv(fpath, index_col="row_id")

        tag         = extract_tag_from_filename(fname, merged=True)
        cutoff_date = cutoff_map.get(tag)
        if cutoff_date is None:
            print(f"  [WARN] No cutoff date for tag '{tag}' — skipping.")
            continue

        output_csv = f"labeled_{fname}"
        label_csv_file(df, output_csv, cutoff_date)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(merge_mode: bool = False) -> None:
    if merge_mode:
        label_merged_datasets()
    else:
        label_multiple_single_datasets()


if __name__ == "__main__":
    # Change to True to label merged (walk-forward training) datasets
    merge_mode = False
    main(merge_mode=merge_mode)
