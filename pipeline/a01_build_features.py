#!/usr/bin/env python3
"""
a01_build_features.py — Step 1: Build feature dataset from raw CSP snapshots.

Loads raw CSV snapshots from a broker (IBKR), merges GEX data, adds VIX and
macro price-return features, and writes the enriched dataset to disk.

The active dataset is selected via `active_process_dataset` in config.yaml.
After build, unique symbols are extracted to a text file so that
a02_collect_events.py can collect the matching corporate events.

Usage:
    python -m pipeline.a01_build_features
    # or from project root:
    python pipeline/a01_build_features.py

Outputs (inside output/data_prep/):
    trades_raw_<tag>.csv              raw merged trades
    trades_with_gex_macro_<tag>.csv   enriched dataset
    output/data_prep/corp_events/symbols_in_option_data_<tag>.txt
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# Ensure project root is on path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service.data_prepare import add_macro_features
from service.preprocess import filter_by_dte, load_csp_files, merge_gex
from service.env_config import get_derived_file, getenv, config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_target_time(s: str) -> time:
    """Parse 'HH:MM' into datetime.time; fallback to 11:00 on error."""
    try:
        hh, mm = s.split(":")
        return time(int(hh), int(mm))
    except Exception:
        return time(11, 0)


@dataclass
class BuildOutputs:
    df: pd.DataFrame
    report: Dict[str, Any]
    paths: Dict[str, Optional[str]]


# ---------------------------------------------------------------------------
# Core builder (importable, no env-variable side-effects)
# ---------------------------------------------------------------------------

def build_dataset_with_features(
    data_dir: str,
    glob_pat: str = "coveredPut_*.csv",
    target_time: str = "11:00",
    gex_base_dir: str = "",
    gex_target_time: str = "11:00",
    vix_csv: Optional[str] = None,
    px_base_dir: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    *,
    enforce_daily_pick: bool = True,
    gex_filter_missing: bool = False,
    out_dir: Optional[str] = None,
    basic_csv_name: Optional[str] = None,
    skip_gex_merge: bool = False,
    filter_func=None,
) -> BuildOutputs:
    """
    Build a feature dataset:
      1. Load raw CSP snapshots (load_csp_files)
      2. Merge GEX data (merge_gex)
      3. Add VIX + macro price-return features (add_macro_features)

    All behavior is controlled by explicit parameters (no env-variable drift).

    Parameters
    ----------
    data_dir : str
        Directory containing raw CSP snapshot CSVs.
    glob_pat : str
        Filename glob pattern for snapshot CSVs.
    target_time : str
        'HH:MM' — daily snapshot selection time.
    gex_base_dir : str
        Base directory for GEX CSV files.
    gex_target_time : str
        'HH:MM' for GEX snapshot selection.
    vix_csv : str, optional
        Path to VIX CSV (None = skip VIX merge).
    px_base_dir : str, optional
        Directory with per-symbol price parquet/csv files.
    enforce_daily_pick : bool
        Keep one snapshot per day at target_time.
    gex_filter_missing : bool
        If True, drop rows where gex_missing == 1.
    out_dir : str, optional
        Write intermediate and final CSVs here.
    basic_csv_name : str, optional
        Used to derive output filenames (via get_derived_file).
    skip_gex_merge : bool
        Skip GEX merge and read a previously written *_gex.csv.
    filter_func : callable, optional
        Applied to raw DataFrame before GEX merge.

    Returns
    -------
    BuildOutputs
    """
    if not gex_base_dir:
        raise ValueError("gex_base_dir must be provided.")

    vix_csv = (vix_csv or "").strip() or None
    px_base_dir = (px_base_dir or "").strip() or None

    gex_t = parse_target_time(gex_target_time)
    gex_target_minutes = gex_t.hour * 60 + gex_t.minute

    # Step 1: Load raw CSP snapshots
    raw = load_csp_files(
        data_dir,
        glob_pat,
        target_time=target_time,
        enforce_daily_pick=enforce_daily_pick,
        start_date=start_date,
        end_date=end_date,
    )
    if filter_func:
        raw = filter_func(raw)
    raw = raw.reset_index().rename(columns={"index": "row_id"})

    paths: Dict[str, Optional[str]] = {"raw_csv": None, "gex_csv": None, "out_csv": None}
    out_dir_path: Optional[Path] = Path(out_dir) if out_dir else None
    if out_dir_path:
        out_dir_path.mkdir(parents=True, exist_ok=True)
        raw_name = os.path.basename(basic_csv_name or "trades_raw_orig.csv")
        raw_csv_path = out_dir_path / raw_name
        raw.to_csv(raw_csv_path, index=False)
        paths["raw_csv"] = str(raw_csv_path)

    # Step 2: Merge GEX
    if skip_gex_merge:
        if not out_dir_path or not paths["raw_csv"]:
            raise ValueError("skip_gex_merge=True requires out_dir and a prior raw_csv.")
        gex_csv_path = out_dir_path / (
            Path(paths["raw_csv"]).stem + "_gex.csv"
        )
        if not gex_csv_path.exists():
            raise FileNotFoundError(f"Expected existing GEX file not found: {gex_csv_path}")
        gex_merged = pd.read_csv(gex_csv_path)
        paths["gex_csv"] = str(gex_csv_path)
    else:
        gex_merged = merge_gex(raw, gex_base_dir, gex_target_minutes)
        if out_dir_path:
            raw_stem = Path(paths["raw_csv"]).stem if paths["raw_csv"] else (
                Path(basic_csv_name or "trades_raw_orig.csv").stem
            )
            gex_csv_path = out_dir_path / f"{raw_stem}_gex.csv"
            gex_merged.to_csv(gex_csv_path, index=False)
            paths["gex_csv"] = str(gex_csv_path)

    # Step 3: Add macro features
    d = add_macro_features(gex_merged, vix_csv, px_base_dir)

    if gex_filter_missing:
        if "gex_missing" not in d.columns:
            raise KeyError("gex_filter_missing=True but 'gex_missing' column is absent.")
        d = d[d["gex_missing"] == 0].copy()

    # Write final enriched CSV
    if out_dir_path and basic_csv_name:
        derived_name = get_derived_file(basic_csv_name)[0] or "dataset_gex_macro.csv"
        out_csv_name = derived_name.replace(".csv", "_gexonly.csv") if gex_filter_missing else derived_name
        out_csv_path = out_dir_path / out_csv_name
        d.to_csv(out_csv_path, index=False)
        paths["out_csv"] = str(out_csv_path)

    report: Dict[str, Any] = {
        "rows_raw": int(len(raw)),
        "rows_gex_merged": int(len(gex_merged)),
        "gex_found": int((gex_merged.get("gex_missing", pd.Series([1] * len(gex_merged))) == 0).sum()),
        "gex_missing": int((gex_merged.get("gex_missing", pd.Series([1] * len(gex_merged))) == 1).sum()),
        "gex_base_dir": gex_base_dir,
        "gex_target_time": gex_target_time,
        "target_time": target_time,
        "enforce_daily_pick": bool(enforce_daily_pick),
        "rows_out": int(len(d)),
        "unique_symbols": int(d["baseSymbol"].nunique()) if "baseSymbol" in d.columns else None,
        "vix_non_null": int(d["VIX"].notna().sum()) if "VIX" in d.columns else None,
        "prev_close_non_null": int(d["prev_close"].notna().sum()) if "prev_close" in d.columns else None,
        "px_base_dir": px_base_dir,
        "vix_csv": vix_csv,
        "gex_filter_missing": bool(gex_filter_missing),
        "paths": paths,
    }

    return BuildOutputs(df=d, report=report, paths=paths)


def extract_and_write_symbols(df: pd.DataFrame, output_path: str) -> int:
    """Write unique baseSymbol values to a text file (one per line).

    Returns the number of symbols written.
    """
    if "baseSymbol" not in df.columns:
        raise ValueError("DataFrame must have 'baseSymbol' column.")
    symbols = sorted(df["baseSymbol"].dropna().unique())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(symbols))
    print(f"[INFO] Wrote {len(symbols)} unique symbols to {output_path}")
    return len(symbols)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Process the dataset for active_process_dataset from config.yaml.

    Incremental mode: if the enriched output CSV already exists, only snapshot
    files whose embedded date is newer than the last captureTime in that CSV are
    loaded and processed (GEX merge + macro features).  New rows are appended to
    the existing CSV so previously computed work is never repeated.
    """
    dataset_cfg = config.get_active_dataset_config()
    if not dataset_cfg:
        raise SystemExit(
            "No dataset configuration found. "
            "Add a `dataset:` block to config.yaml."
        )

    print(f"[INFO] Data directory: {dataset_cfg.get('data_dir')}")

    data_dir     = dataset_cfg.get("data_dir", "")
    basic_csv    = dataset_cfg.get("data_basic_csv", "trades_raw_orig.csv")
    symbols_file = dataset_cfg.get("tickers_file", "")
    start_date   = dataset_cfg.get("events_start_date", "")
    end_date     = dataset_cfg.get("events_end_date", "")

    if not data_dir:
        raise SystemExit("data_dir not specified in dataset configuration.")

    glob_pat        = getenv("DATA_GLOB", "coveredPut_*.csv")
    target_time     = getenv("DATA_TARGET_TIME", "11:00")
    out_dir         = os.path.join(getenv("COMMON_OUTPUT_DIR", "output"), "data_prep")
    base_dir        = getenv("GEX_BASE_DIR")
    gex_target_time = getenv("GEX_TARGET_TIME", "11:00")

    if not base_dir:
        raise SystemExit("GEX_BASE_DIR is not set in config.yaml or .env")

    vix_csv     = getenv("MACRO_VIX_CSV", "").strip() or None
    px_base_dir = getenv("MACRO_PX_BASE_DIR", "").strip() or None

    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Incremental: find the last snapshot date already in the enriched CSV
    # so we skip re-loading + re-merging everything that's already done.
    # ------------------------------------------------------------------
    derived_macro_name = get_derived_file(basic_csv)[0]
    enriched_path = os.path.join(out_dir, derived_macro_name) if derived_macro_name else None

    effective_start = start_date or None
    is_incremental  = False
    if enriched_path and os.path.isfile(enriched_path):
        try:
            ct = pd.read_csv(enriched_path, usecols=["captureTime"])
            ct["captureTime"] = pd.to_datetime(ct["captureTime"], errors="coerce")
            last = ct["captureTime"].dt.normalize().max()
            if pd.notna(last):
                candidate = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                if effective_start is None or pd.Timestamp(candidate) > pd.Timestamp(effective_start):
                    effective_start = candidate
                    is_incremental  = True
                    print(f"[INFO] Incremental mode: enriched CSV covers up to {last.date()}, "
                          f"loading new snapshots from {effective_start} onward.")
        except Exception as exc:
            print(f"[WARN] Could not inspect existing enriched CSV ({exc}); full rebuild.")

    if is_incremental and end_date and pd.Timestamp(effective_start) > pd.Timestamp(end_date):
        print(f"[INFO] Already up to date through {end_date}. Nothing to process.")
        if symbols_file and enriched_path and os.path.isfile(enriched_path):
            sym_df = pd.read_csv(enriched_path, usecols=["baseSymbol"])
            extract_and_write_symbols(sym_df, symbols_file)
        return

    print(f"[INFO] Building dataset with features "
          f"({'incremental' if is_incremental else 'full'})…")

    # Pass out_dir=None so build_dataset_with_features does not write files;
    # we control writing below to support append mode.
    out = build_dataset_with_features(
        data_dir=data_dir,
        glob_pat=glob_pat,
        target_time=target_time,
        gex_base_dir=base_dir,
        gex_target_time=gex_target_time,
        vix_csv=vix_csv,
        px_base_dir=px_base_dir,
        start_date=effective_start,
        end_date=end_date or None,
        enforce_daily_pick=False,
        gex_filter_missing=False,
        out_dir=None,
        basic_csv_name=None,
        filter_func=filter_by_dte,
    )
    print(json.dumps(out.report, indent=2))

    new_df = out.df
    if new_df.empty:
        print("[INFO] No new rows after DTE filter. Enriched CSV unchanged.")
        if symbols_file and enriched_path and os.path.isfile(enriched_path):
            sym_df = pd.read_csv(enriched_path, usecols=["baseSymbol"])
            extract_and_write_symbols(sym_df, symbols_file)
        return

    # Write (fresh) or append (incremental) enriched CSV
    if is_incremental and enriched_path and os.path.isfile(enriched_path):
        new_df.to_csv(enriched_path, mode="a", header=False, index=False)
        print(f"[INFO] Appended {len(new_df):,} new rows → {enriched_path}")
    else:
        new_df.to_csv(enriched_path, index=False)
        print(f"[INFO] Wrote {len(new_df):,} rows → {enriched_path}")

    # Update symbols file from the full (cumulative) enriched dataset
    if symbols_file:
        sym_df = pd.read_csv(enriched_path, usecols=["baseSymbol"]) if is_incremental else new_df
        extract_and_write_symbols(sym_df, symbols_file)
        print("[SUCCESS] Ready for a02_collect_events.py")
    else:
        print("[WARN] No tickers_file specified — skipping symbol extraction.")

    # Write build log
    ts = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = Path("log") / f"a01_build_log_{ts}.json"
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({**out.report, "incremental": is_incremental,
                   "effective_start": effective_start}, f, indent=2)


if __name__ == "__main__":
    main()
