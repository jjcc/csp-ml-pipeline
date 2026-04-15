#!/usr/bin/env python3
"""
a05_merge_datasets.py — Step 5: Build a rolling-window training dataset.

Selects the last `rolling_window_weeks` weeks of labeled batches that precede
the active_score_dataset, merges them into a single CSV, and writes it to
output/data_merged/.

Output filename convention
--------------------------
    merged_roll{W}w_{YYYYMMDD}.csv

where W is rolling_window_weeks and YYYYMMDD is the events_start_date of the
active_score_dataset.  Example: merged_roll14w_20251027.csv

This file is the direct input to b01_train_winner.py.

Rolling window selection
------------------------
    window_end   = events_start_date of active_score_dataset
    window_start = window_end − rolling_window_weeks

All dataset configs whose events_start_date falls in [window_start, window_end)
are included.  The score dataset itself is always excluded.

Usage:
    python pipeline/a05_merge_datasets.py
"""

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service.env_config import config, get_derived_file, getenv
from service.constants import MAX_DAYS_TO_EXPIRATION


def _labeled_csv_path(cfg: dict, prep_dir: str) -> str:
    """Return the absolute path to the labeled macro CSV for a dataset config."""
    basic_csv = cfg.get("data_basic_csv", "")
    macro_csv, _ = get_derived_file(basic_csv)
    if macro_csv is None:
        raise ValueError(f"Cannot derive macro CSV path from data_basic_csv={basic_csv!r}")
    return os.path.join(prep_dir, os.path.basename(macro_csv))


def main() -> None:
    output_dir = getenv("COMMON_OUTPUT_DIR", "output")
    prep_dir   = os.path.join(output_dir, "data_prep")
    merge_dir  = os.path.join(output_dir, "data_merged")
    os.makedirs(merge_dir, exist_ok=True)

    # --- Resolve rolling window batches ---
    window_weeks = int(getenv("ROLLING_WINDOW_WEEKS", "14"))
    batches      = config.get_rolling_train_batches()

    if not batches:
        print(
            "[ERROR] No training batches found for the rolling window.\n"
            "  Check that active_score_dataset is set and common_configs contains\n"
            "  at least one batch whose events_start_date falls within the window."
        )
        sys.exit(1)

    score_date    = config.get_score_date()          # e.g. "20251027"
    score_cfg     = config.get_score_dataset_config()
    score_tag     = score_cfg.get("data_basic_csv", "?")

    print(f"Rolling window: {window_weeks} weeks before {score_date} (score dataset: {score_tag})")
    print(f"  → {len(batches)} training batch(es) selected:")
    for b in batches:
        print(f"     {b['data_basic_csv']}  [{b['events_start_date']} – {b['events_end_date']}]")

    # --- Load and concatenate ---
    dfs = []
    for b in batches:
        fpath = _labeled_csv_path(b, prep_dir)
        if not os.path.isfile(fpath):
            print(f"  [WARN] Missing: {fpath} — skipping this batch")
            continue
        df_i = pd.read_csv(fpath)
        dfs.append(df_i)
        print(f"  Loaded {len(df_i):>6,} rows  ← {os.path.basename(fpath)}")

    if not dfs:
        print("[ERROR] All batch files are missing.  Run a01–a04 for each batch first.")
        sys.exit(1)

    merged = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total rows before filters : {len(merged):,}")

    # --- Filters (same as original pipeline) ---
    if "daysToExpiration" in merged.columns:
        merged = merged[merged["daysToExpiration"] <= MAX_DAYS_TO_EXPIRATION]
        print(f"  Rows after DTE ≤ {MAX_DAYS_TO_EXPIRATION} filter : {len(merged):,}")

    # Apply cutoff_date from the *last* batch in the window (most recent)
    last_batch   = batches[-1]
    cutoff_date  = last_batch.get("cutoff_date")
    if cutoff_date:
        before = len(merged)
        merged = merged[
            pd.to_datetime(merged["tradeTime"], errors="coerce")
            <= pd.to_datetime(cutoff_date)
        ]
        print(f"  Rows after cutoff {cutoff_date}  : {len(merged):,}  (removed {before - len(merged):,})")

    # --- Write output ---
    out_name = f"merged_roll{window_weeks}w_{score_date}.csv"
    out_path = os.path.join(merge_dir, out_name)
    merged.to_csv(out_path, index=False)

    print(f"\n✅  Merged dataset written → {out_path}")
    print(f"   {len(merged):,} rows  |  {len(batches)} batches  |  window = {window_weeks} weeks")


if __name__ == "__main__":
    main()
