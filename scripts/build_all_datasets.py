#!/usr/bin/env python3
"""
Build all datasets with features (batch processing).

This script processes ALL datasets defined in common_configs.
For single-dataset processing controlled by active_process_dataset,
use a00build_dataset_with_features.py instead.

Usage:
    python scripts/build_all_datasets.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

from a00build_dataset_with_features import (
    build_dataset_with_features,
    extract_and_write_symbols,
    filter_by_dte,
)
from service.env_config import config, getenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main():
    """Process all datasets in common_configs."""
    # Pipeline configuration
    glob_pat = getenv("DATA_GLOB", "coveredPut_*.csv")
    target_time = getenv("DATA_TARGET_TIME", "11:00")

    # Outputs
    out_dir = getenv("COMMON_OUTPUT_DIR", "output")
    out_dir = os.path.join(out_dir, "data_prep")
    os.makedirs(out_dir, exist_ok=True)

    # GEX source
    base_dir = getenv("GEX_BASE_DIR")
    gex_target_time_str = getenv("GEX_TARGET_TIME", "11:00")
    if not base_dir:
        raise SystemExit("GEX_BASE_DIR is not set in config.yaml or .env")

    # VIX and price sources
    VIX_CSV = getenv("MACRO_VIX_CSV", "").strip() or None
    PX_BASE_DIR = getenv("MACRO_PX_BASE_DIR", "").strip() or None

    gex_filter_missing = False
    ENFORCE_DAILY_PICK = False

    # Process all datasets
    common_configs = config.get_common_configs_raw()
    processed = []
    skipped = []

    for k, v in common_configs.items():
        basic_csv = v.get("data_basic_csv", "trades_raw_orig.csv")
        data_dir_k = v.get("data_dir", "")
        symbols_file = v.get("tickers_file", "")

        if k == "original":
            print(f"\n[SKIP] Skipping {k}")
            skipped.append(k)
            continue

        print(f"\n{'='*60}")
        print(f"[INFO] Processing config: {k}")
        print(f"[INFO] Data directory: {data_dir_k}")
        print(f"{'='*60}")

        try:
            out = build_dataset_with_features(
                data_dir=data_dir_k,
                glob_pat=glob_pat,
                target_time=target_time,
                gex_base_dir=base_dir,
                gex_target_time=gex_target_time_str,
                vix_csv=VIX_CSV,
                px_base_dir=PX_BASE_DIR,
                enforce_daily_pick=ENFORCE_DAILY_PICK,
                gex_filter_missing=gex_filter_missing,
                out_dir=out_dir,
                basic_csv_name=basic_csv,
                filter_func=filter_by_dte,
            )

            # Print build report
            print(json.dumps(out.report, indent=2))

            # Write log file
            date_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            log_fn = os.path.join(
                "log",
                f"a00build_dataset_log_{k}_{date_str}.json"
                .replace(" ", "_")
                .replace(":", "-")
            )
            os.makedirs("log", exist_ok=True)
            with open(log_fn, "w") as f:
                json.dump(out.report, f, indent=2)

            # Extract symbols
            if symbols_file:
                num_symbols = extract_and_write_symbols(out.df, symbols_file)
                print(f"[SUCCESS] Wrote {num_symbols} symbols to {symbols_file}")

            processed.append(k)

        except Exception as e:
            print(f"[ERROR] Failed to process {k}: {e}")
            skipped.append(k)

    # Summary
    print(f"\n{'='*60}")
    print(f"[SUMMARY] Batch processing complete")
    print(f"  Processed: {len(processed)} datasets - {processed}")
    print(f"  Skipped: {len(skipped)} datasets - {skipped}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
