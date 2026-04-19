#!/usr/bin/env python3
"""
model_utils.py — Shared utilities for the CSP pipeline.

Contains:
- ensure_dir(path): create parent dirs and return path
- env helpers: load_env_default()
- Data prep:
    * prep_tail_like_training(df, feats, medians): uses training medians; fills gex_missing=1
    * prep_winner_like_training(df, feats, medians=None, impute_missing=True): impute or drop-NaN to match training
- Threshold selection:
    * pick_threshold_auto(y_true, proba, targets_prec=None, targets_rec=None)
      - precision target: choose lowest threshold reaching >= target precision
      - recall target: choose highest threshold reaching >= target recall
Additions:
- prep_tail_training_derived(df): EXACTLY replicates training's _prep_df()
  (recompute total_pnl and return_pct; parse datetimes; bid/bidPrice handling)
- fill_features_with_training_medians(df, feat_list, medians): EXACTLY replicates training's _fill_features()
  (create missing cols; gex_missing:=1 when NaN; other NaNs -> training medians)
"""

import glob
import os
from pathlib import Path
from time import sleep
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score

from service.constants import BASE_FEATS, GEX_FEATS, NEW_FEATS, ALL_FEATS, DEFAULT_SLEEP_SECONDS

# Magic numbers replaced by constants

# ---------- FS / ENV helpers ----------

# ---- Existing helpers kept minimal to avoid conflicts ----
def ensure_dir(path: str) -> str:
    p = Path(path)
    target = p if not p.suffix else p.parent
    target.mkdir(parents=True, exist_ok=True)
    return path

def load_env_default() -> None:
    """Try to load .env from CWD; if missing, try alongside the calling script."""
    loaded = load_dotenv()
    if not loaded:
        here = Path(__file__).resolve().parent / ".env"
        if here.exists():
            load_dotenv(dotenv_path=here)


# ---------- Data preparation ----------
def prep_tail_training_df(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce train_tail_with_gex._prep_df exactly."""
    X = df.copy()

    # Parse datetimes if present
    for c in ("tradeTime","expirationDate"):
        if c in X.columns:
            X[c] = pd.to_datetime(X[c], errors="coerce")

    # Recompute total_pnl and return_pct the same way as trainer
    # If 'bid' missing but 'bidPrice' present, alias it
    if "bid" not in X.columns and "bidPrice" in X.columns:
        X["bid"] = X["bidPrice"]

    bid_eff = X["bid"] if "bid" in X.columns else X.get("bidPrice", pd.Series(index=X.index, dtype=float))

    X["entry_credit"] = pd.to_numeric(bid_eff, errors="coerce").fillna(0.0) * 100.0

    strike = pd.to_numeric(X["strike"], errors="coerce")
    expiry_close = pd.to_numeric(X.get("expiry_close"), errors="coerce")
    X["exit_intrinsic"] = np.maximum(strike - expiry_close, 0.0) * 100.0

    X["total_pnl"] = X["entry_credit"] - X["exit_intrinsic"]

    capital = pd.to_numeric(X["strike"], errors="coerce") * 100.0
    X["return_pct"] = 100.0 * X["total_pnl"] / capital

    return X

# Alias for backward compatibility
prep_tail_training_derived = prep_tail_training_df

def fill_features_with_training_medians(df: pd.DataFrame, feat_list: List[str], medians: Optional[Dict[str, float]] = None):
    """
    Reproduce train_tail_with_gex._fill_features for scoring:
    - Ensure all features exist
    - For 'gex_missing': fillna(1)
    - For all others: fillna(training_median)

    If medians is None, computes them from df and returns (X, medians).
    If medians is provided, uses them and returns X only.
    """
    X = df.copy()

    if medians is None:
        # Compute medians from data (training mode)
        medians_x = {}
        for c in feat_list:
            if c not in X.columns:
                X[c] = np.nan
            if c == "gex_missing":
                X[c] = X[c].fillna(1)
                medians_x[c] = 0.0
            else:
                medx = X[c].median(skipna=True)
                medians_x[c] = float(medx) if pd.notna(medx) else 0.0
                X[c] = X[c].fillna(medians_x[c])
        return X[feat_list].astype(float), medians_x
    else:
        # Use provided medians (scoring mode)
        for c in feat_list:
            if c not in X.columns:
                X[c] = np.nan
            if c == "gex_missing":
                X[c] = X[c].fillna(1)
            else:
                med = float(medians.get(c, 0.0))
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(med)
        return X[feat_list].astype(float)


def prep_winner_like_training(
    df: pd.DataFrame,
    feats: List[str],
    medians: Optional[Dict[str, float]] = None,
    impute_missing: bool = True,
):
    """
    Prepare features like winner training.
    Returns (X_prepared, mask_kept_rows). If impute_missing=False, drops rows with any NaNs.
    """
    if impute_missing:
        X = df.copy()
        for c in feats:
            if c not in X.columns:
                X[c] = np.nan
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna((medians or {}).get(c, 0.0))
        mask = pd.Series(True, index=df.index)
        return X[feats], mask
    else:
        X = df[feats].apply(pd.to_numeric, errors="coerce")
        mask = (~X.isna().any(axis=1))
        return X[mask], mask


# ---------- Threshold selection ----------

def pick_threshold_auto(
    y_true: np.ndarray,
    proba: np.ndarray,
    targets_prec: Optional[List[float]] = None,
    targets_rec: Optional[List[float]] = None,
):
    """Return (chosen_threshold, pandas.DataFrame row) according to constraints.
       - Precision target: choose the *lowest threshold* achieving >= first target precision.
       - Recall target: choose the *highest threshold* achieving >= first target recall.
       If neither applies, returns (None, None).
    """
    import pandas as pd  # local import to avoid global dependency here

    if targets_prec:
        targets_prec = [float(targets_prec[0])]
    if targets_rec:
        targets_rec = [float(targets_rec[0])]

    if not targets_prec and not targets_rec:
        return None, None

    precision, recall, thresholds = precision_recall_curve(y_true, proba)

    def metrics_at(thr: float):
        yhat = (proba >= thr).astype(int)
        return dict(
            precision=precision_score(y_true, yhat, zero_division=0),
            recall=recall_score(y_true, yhat, zero_division=0),
            f1=f1_score(y_true, yhat, zero_division=0),
            coverage=float(yhat.mean()),
            threshold=float(thr),
        )

    if targets_prec:
        t = targets_prec[0]
        chosen = None
        for thr in sorted(thresholds):
            m = metrics_at(thr)
            if m["precision"] >= t:
                chosen = m; break
        if chosen is None and len(thresholds):
            chosen = metrics_at(float(max(thresholds)))
        row = pd.DataFrame([{**chosen, "target_type": "precision", "target": float(t)}])
        return chosen["threshold"], row

    if targets_rec:
        t = targets_rec[0]
        chosen = None
        for thr in sorted(thresholds, reverse=True):
            m = metrics_at(thr)
            if m["recall"] >= t:
                chosen = m
        if chosen is None and len(thresholds):
            chosen = metrics_at(float(min(thresholds)))
        row = pd.DataFrame([{**chosen, "target_type": "recall", "target": float(t)}])
        return chosen["threshold"], row

    return None, None

def get_features_rg(df: pd.DataFrame, feature_list: str):
    if not feature_list or feature_list.strip() == "":
        default = ALL_FEATS
        feats = [c for c in default if c in df.columns]
    else:
        feats = [c.strip() for c in feature_list.split(",") if c.strip() in df.columns]
        if not feats:
            raise ValueError("No valid features from FEATURES env var found in CSV.")
    return feats


def get_symbols_last_few_days(folder, end_date, days=5):
    days_including_wkd = days + 2
    dates = pd.date_range(end=end_date, periods=days_including_wkd).to_pydatetime().tolist()
    dates = [d for d in dates if d.weekday() < 5]
    date_strs = [d.strftime("%Y-%m-%d") for d in dates ]
    files = []
    for d in date_strs:
        pattern = f"coveredPut_{d}*.csv"
        paths = glob.glob(os.path.join(folder, pattern))
        files.extend(paths)
        files = sorted(files)
    if len(files) == 0:
        print(f"No files found for last 5 trading days up to {end_date}.")
        return
    df = pd.DataFrame()
    for f in files:
        temp_df = pd.read_csv(f)
        df = pd.concat([df, temp_df], ignore_index=True)
    symbols = df['baseSymbol'].unique().tolist()
    return files,symbols


def download_prices_batched(symbols, start_dt, end_dt, batch_size=30, threads=True):
    """
    Download daily OHLCV prices for many symbols in batches using yfinance.
    Returns dict: symbol -> pd.DataFrame (Open, High, Low, Close, Volume, etc.)
    """
    import yfinance as yf
    import math
    load_dotenv(override=False)
    sleep_seconds = float(os.getenv("YF_HISTORY_SLEEP_SECONDS", str(DEFAULT_SLEEP_SECONDS)))
    threads_override = os.getenv("YF_HISTORY_THREADS", "").strip().lower()
    if threads_override:
        threads = threads_override in {"1", "true", "yes", "y", "on"}
    symbols = sorted(set([s for s in symbols if isinstance(s, str) and len(s)>0]))
    symbols = [s if "." not in s else s.replace(".", "-") for s in symbols]  # ensure uppercase
    out = {}
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        try:
            df = yf.download(batch, start=start_dt.date(), end=(end_dt + pd.Timedelta(days=1)).date(),
                             interval="1d", group_by="ticker", auto_adjust=False, threads=threads, progress=False)
            # yfinance returns multi-index columns when multiple tickers
            if isinstance(df.columns, pd.MultiIndex):
                # df columns like ('AAPL','Open'), ('AAPL','High'), ('AAPL','Low'), ('AAPL','Close'), etc.
                for sym in batch:
                    sym_cols = [col for col in df.columns if col[0] == sym]
                    if sym_cols:
                        sym_df = df[[col for col in sym_cols]]
                        # Flatten column names: ('AAPL', 'Close') -> 'Close'
                        sym_df.columns = [col[1] for col in sym_df.columns]
                        sym_df = sym_df.dropna(how='all')
                        sym_df.index = pd.to_datetime(sym_df.index)
                        out[sym] = sym_df
            else:
                # single ticker case - df already has columns like 'Open', 'High', 'Low', 'Close', etc.
                df = df.dropna(how='all')
                df.index = pd.to_datetime(df.index)
                out[batch[0]] = df
        except Exception as e:
            print(f"[WARN] batch download failed for {batch}: {e}")
        if sleep_seconds > 0 and i + batch_size < len(symbols):
            sleep(sleep_seconds)
    return out
