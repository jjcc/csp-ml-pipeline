#!/usr/bin/env python3
"""
a04_label_data.py — Step 4: Label option trades with win/loss outcomes.

Fetches expiry-date closing prices via yfinance (with caching), computes PnL
and return metrics, and writes a labeled CSV ready for model training.

Symbol exclusions are loaded from data files:
  - data/missing_stocks.json  — symbols with no price data (auto-managed)
  - data/exclude_stocks.json  — manually curated exclusions

Usage:
    python pipeline/a04_label_data.py

Configuration:
    Reads from the single `dataset:` block in config.yaml.
    cutoff_date and output_csv come from that block.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

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

    # ------------------------------------------------------------------
    # Normalised return metrics (matches a09label_data.py in csp_feature_lab2)
    # These are also computed by service/preprocess.py::add_dte_and_normalized_returns()
    # at training time, but including them here makes the labeled CSV self-contained
    # for diagnostics and for the tail classifier OOF output (which logs return_mon).
    # IMPORTANT: return_mon must use the same column as WINNER_TRAIN_TARGET (default "return_mon").
    # ------------------------------------------------------------------
    dte = pd.to_numeric(df.get("daysToExpiration", pd.Series(1, index=df.index)),
                        errors="coerce").replace(0, 1)
    df["return_per_day"] = df["return_pct"] / dte
    df["return_ann"]     = df["return_pct"] * 365.0 / dte
    df["return_mon"]     = df["return_pct"] * 30.0  / dte  # primary training target

    # ------------------------------------------------------------------
    # Pre-compute 4-bin per-day quartile labels (cross-sectional ranking)
    # Bins are computed from return_mon to match WINNER_TRAIN_TARGET.
    # This mirrors the assign_bins() logic in csp_feature_lab2/a09label_data.py
    # (bins4_quick_fixes.md §Issue 1: label/target alignment).
    # Note: b01_train_winner.py recomputes bins dynamically via build_label_bins4();
    # y_bin here is purely for analysis / early diagnostics.
    # ------------------------------------------------------------------
    df["_trade_date"] = pd.to_datetime(df["tradeTime"], errors="coerce").dt.tz_localize(None).dt.normalize()

    def _assign_bins(g: pd.DataFrame) -> pd.DataFrame:
        s = g["return_mon"]
        if s.notna().sum() < 20:
            g["y_bin"] = np.nan
            return g
        q25, q50, q75 = s.quantile([0.25, 0.50, 0.75]).values

        def _to_bin(x):
            if not np.isfinite(x):
                return np.nan
            if x <= q25: return 0
            if x <= q50: return 1
            if x <= q75: return 2
            return 3

        g["y_bin"] = s.apply(_to_bin)
        return g

    df = df.groupby("_trade_date", group_keys=False).apply(_assign_bins)
    df.drop(columns=["_trade_date"], inplace=True, errors="ignore")

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
# Single-dataset labeling
# ---------------------------------------------------------------------------

def label_single_dataset() -> None:
    """Label the corporate-event-filtered dataset produced by a03_filter_trades.py.

    Reads dataset.filtered_trades_csv (the a03 output) so that corporate-event
    filtering is honoured.  Falls back to the enriched macro CSV if the filtered
    file is not found (with a warning), so the step can still run if a02/a03
    were skipped for testing purposes.

    Writes the labeled CSV to output/data_labeled/<dataset.output_csv>.
    """
    from service.env_config import config, get_derived_file

    ds_cfg     = config.get_active_dataset_config()
    if not ds_cfg:
        raise SystemExit(
            "No dataset configuration found. "
            "Add a `dataset:` block to config.yaml."
        )

    cutoff_date      = ds_cfg.get("cutoff_date")
    output_csv       = ds_cfg.get("output_csv", "")
    filtered_csv     = ds_cfg.get("filtered_trades_csv", "")

    if not cutoff_date:
        raise SystemExit("dataset.cutoff_date is not set in config.yaml.")
    if not output_csv:
        raise SystemExit("dataset.output_csv is not set in config.yaml.")

    # Prefer the filtered output from a03 (corporate-event filtering applied).
    # Fall back to the enriched macro CSV only if filtered file is absent.
    if filtered_csv and os.path.isfile(filtered_csv):
        fpath = filtered_csv
        print(f"\nProcessing (filtered): {filtered_csv}")
        df = pd.read_csv(fpath)
    else:
        if filtered_csv:
            print(
                f"[WARN] Filtered CSV not found: {filtered_csv}\n"
                f"       Run a03_filter_trades.py to apply corporate-event filtering.\n"
                f"       Falling back to enriched macro CSV (no event filtering)."
            )
        basic_csv = ds_cfg.get("data_basic_csv", "")
        macro_csv, _ = get_derived_file(basic_csv)
        if not macro_csv:
            raise SystemExit(
                f"Cannot derive enriched CSV name from data_basic_csv={basic_csv!r}"
            )
        input_dir = os.path.join(getenv("COMMON_OUTPUT_DIR", "output"), "data_prep")
        fpath     = os.path.join(input_dir, macro_csv)
        if not os.path.isfile(fpath):
            raise SystemExit(
                f"Enriched CSV not found: {fpath}\n"
                f"  Run a01_build_features.py first."
            )
        print(f"\nProcessing (unfiltered): {macro_csv}")
        df = pd.read_csv(fpath, index_col="row_id")

    exclude = load_exclude_symbols()
    before  = len(df)
    df = df[~df["baseSymbol"].isin(exclude)].copy()
    if len(df) != before:
        print(f"  Excluded {before - len(df)} rows from known bad symbols.")

    label_csv_file(df, output_csv, cutoff_date)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    label_single_dataset()


if __name__ == "__main__":
    main()
