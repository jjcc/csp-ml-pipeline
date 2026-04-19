#!/usr/bin/env python3
"""
a05_merge_datasets.py — Step 5: Build a rolling-window training dataset.

Loads the single labeled dataset (produced by a04_label_data.py) and
date-filters it to the rolling training window that precedes the scoring
period defined in config.yaml.

Output filename convention
--------------------------
    merged_roll{W}w_{YYYYMMDD}.csv

where W is rolling_window_weeks and YYYYMMDD is dataset.events_start_date.
Example: merged_roll14w_20251027.csv

This file is the direct input to b01_train_winner.py.

Rolling window selection
------------------------
    window_end   = dataset.events_start_date   (= start of the scoring period)
    window_start = window_end − rolling_window_weeks

Only rows whose tradeTime falls in [window_start, window_end) are kept.
This naturally excludes the scoring period from the training set.

Usage:
    python pipeline/a05_merge_datasets.py
"""

import os
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service.env_config import config, getenv
from service.constants import MAX_DAYS_TO_EXPIRATION


def main() -> None:
    output_dir  = getenv("COMMON_OUTPUT_DIR", "output")
    labeled_dir = os.path.join(output_dir, "data_labeled")
    merge_dir   = os.path.join(output_dir, "data_merged")
    os.makedirs(merge_dir, exist_ok=True)

    # --- Resolve dataset config and rolling window ---
    window_weeks = int(getenv("ROLLING_WINDOW_WEEKS", "14"))
    ds_cfg       = config.get_score_dataset_config()
    score_date   = config.get_score_date()   # YYYYMMDD, e.g. "20251027"

    if not ds_cfg:
        print("[ERROR] No dataset config found in config.yaml. Add a `dataset:` block.")
        sys.exit(1)

    score_start_str = ds_cfg.get("events_start_date", "")
    output_csv      = ds_cfg.get("output_csv", "")

    if not score_start_str:
        print("[ERROR] dataset.events_start_date is not set in config.yaml.")
        sys.exit(1)
    if not output_csv:
        print("[ERROR] dataset.output_csv is not set in config.yaml.")
        sys.exit(1)

    score_start  = pd.Timestamp(score_start_str).normalize()
    window_start = score_start - timedelta(weeks=window_weeks)

    labeled_path = os.path.join(labeled_dir, output_csv)
    if not os.path.isfile(labeled_path):
        print(
            f"[ERROR] Labeled CSV not found: {labeled_path}\n"
            f"  Run a01 → a04 first to produce the labeled dataset."
        )
        sys.exit(1)

    print(f"Loading labeled data: {labeled_path}")
    df = pd.read_csv(labeled_path)
    print(f"  Total rows in labeled CSV: {len(df):,}")

    # --- Date-filter to rolling training window ---
    if "tradeTime" not in df.columns:
        print("[ERROR] 'tradeTime' column not found in labeled CSV.")
        sys.exit(1)

    tt = pd.to_datetime(df["tradeTime"], errors="coerce")
    mask = (tt >= window_start) & (tt < score_start)
    train = df[mask].copy()

    print(f"\nRolling window : {window_start.date()} ≤ tradeTime < {score_start.date()}"
          f"  ({window_weeks} weeks)")
    print(f"  Rows in window : {len(train):,}  "
          f"(excluded {len(df) - int(mask.sum()):,} outside window)")

    if train.empty:
        print(
            "[ERROR] No training rows found in the rolling window.\n"
            "  Check that dataset.events_start_date is correct and that the labeled CSV\n"
            "  contains trades from the preceding period."
        )
        sys.exit(1)

    # --- DTE filter ---
    if "daysToExpiration" in train.columns:
        before = len(train)
        train = train[train["daysToExpiration"] <= MAX_DAYS_TO_EXPIRATION]
        removed = before - len(train)
        if removed:
            print(f"  After DTE ≤ {MAX_DAYS_TO_EXPIRATION} filter : {len(train):,}  (removed {removed:,})")

    # --- Write output ---
    out_name = f"merged_roll{window_weeks}w_{score_date}.csv"
    out_path = os.path.join(merge_dir, out_name)
    train.to_csv(out_path, index=False)

    print(f"\n✅  Training set written → {out_path}")
    print(f"   {len(train):,} rows  |  rolling window = {window_weeks} weeks")


if __name__ == "__main__":
    main()
