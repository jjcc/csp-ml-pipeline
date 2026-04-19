#!/usr/bin/env python3
"""
a03_filter_trades.py — Step 3: Filter option trades near corporate events to remove noise.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

import pandas as pd
import numpy as np

# Ensure project root is on path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class ExclusionWindow:
    """Exclusion window configuration for an event type"""
    days_before_trade: int
    days_after_trade: int
    days_before_expiry: int
    days_after_expiry: int


@dataclass
class FilterConfig:
    """Configuration for trade filtering"""
    trades_csv: str
    events_csv: str
    output_csv: str
    exclusion_windows: Dict[str, ExclusionWindow]
    symbol_col: str = "baseSymbol"
    trade_date_col: str = "tradeTime"
    expiry_col: str = "expirationDate"
    keep_filtered_trades: bool = False  # If True, output filtered trades separately
    filtered_csv: str = ""  # Where to save filtered-out trades


def load_config(config_path: str = "corp_action_config.yaml") -> FilterConfig:
    """Load configuration from YAML file and config.yaml"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load common settings from corp_action_config.yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Load dataset-specific settings from config.yaml
    from service.env_config import config as env_config
    dataset_cfg = env_config.get_active_dataset_config()

    if not dataset_cfg:
        raise SystemExit(
            "No dataset configuration found. "
            "Add a `dataset:` block to config.yaml."
        )

    # Parse exclusion windows from corp_action_config.yaml
    windows = {}
    exclusion_cfg = cfg.get("exclusion_windows", {})

    for event_type, settings in exclusion_cfg.items():
        windows[event_type.upper()] = ExclusionWindow(
            days_before_trade=settings.get("days_before_trade", 0),
            days_after_trade=settings.get("days_after_trade", 0),
            days_before_expiry=settings.get("days_before_expiry", 0),
            days_after_expiry=settings.get("days_after_expiry", 0),
        )

    # Build trades path: a01 writes the enriched CSV to output/data_prep/,
    # not to the raw data_dir.  Derive the macro CSV name from data_basic_csv.
    from service.env_config import get_derived_file, getenv as _getenv
    basic_csv   = dataset_cfg.get("data_basic_csv", "")
    macro_csv, _ = get_derived_file(basic_csv)
    if not macro_csv:
        raise SystemExit(
            f"Cannot derive enriched CSV name from data_basic_csv={basic_csv!r}. "
            "Check the dataset.data_basic_csv field in config.yaml."
        )
    prep_dir     = os.path.join(_getenv("COMMON_OUTPUT_DIR", "output"), "data_prep")
    trades_input = os.path.join(prep_dir, macro_csv)

    return FilterConfig(
        trades_csv=trades_input,
        events_csv=dataset_cfg.get("events_output", ""),  # output from a01
        output_csv=dataset_cfg.get("filtered_trades_csv", ""),
        exclusion_windows=windows,
        symbol_col=cfg.get("symbol_col", "baseSymbol"),
        trade_date_col=cfg.get("trade_date_col", "tradeTime"),
        expiry_col=cfg.get("expiry_col", "expirationDate"),
        keep_filtered_trades=cfg.get("keep_filtered_trades", False),
        filtered_csv=dataset_cfg.get("filtered_out_csv", ""),
    )


def load_trades(csv_path: str, symbol_col: str, trade_date_col: str, expiry_col: str) -> pd.DataFrame:
    """Load and prepare trades CSV"""
    print(f"Loading trades from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    if symbol_col not in df.columns:
        raise ValueError(f"Symbol column '{symbol_col}' not found in trades CSV")
    if trade_date_col not in df.columns:
        raise ValueError(f"Trade date column '{trade_date_col}' not found in trades CSV")
    if expiry_col not in df.columns:
        raise ValueError(f"Expiry column '{expiry_col}' not found in trades CSV")

    # Parse dates
    df[trade_date_col] = pd.to_datetime(df[trade_date_col], errors="coerce")
    df[expiry_col] = pd.to_datetime(df[expiry_col], errors="coerce")

    # Normalize to date only (remove time component)
    df["_trade_date"] = df[trade_date_col].dt.normalize()
    df["_expiry_date"] = df[expiry_col].dt.normalize()

    # Uppercase symbols
    df["_symbol"] = df[symbol_col].astype(str).str.upper().str.strip()

    # Drop rows with missing critical data
    initial_count = len(df)
    df = df.dropna(subset=["_trade_date", "_expiry_date", "_symbol"])
    dropped = initial_count - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with missing dates/symbols")

    print(f"  Loaded {len(df)} trades")
    return df


def load_events(csv_path: str) -> pd.DataFrame:
    """Load and prepare corporate events CSV"""
    print(f"Loading events from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    if "ticker" not in df.columns:
        raise ValueError("Events CSV must have 'ticker' column")
    if "event_type" not in df.columns:
        raise ValueError("Events CSV must have 'event_type' column")
    if "event_date" not in df.columns:
        raise ValueError("Events CSV must have 'event_date' column")

    # Parse dates and normalize
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date"])
    df["_event_date"] = df["event_date"].dt.normalize()

    # Uppercase symbols and event types
    df["_symbol"] = df["ticker"].astype(str).str.upper().str.strip()
    df["_event_type"] = df["event_type"].astype(str).str.upper().str.strip()

    print(f"  Loaded {len(df)} events")
    print(f"  Event types: {df['_event_type'].value_counts().to_dict()}")
    return df


def find_nearest_events(
    trades_df: pd.DataFrame,
    events_df: pd.DataFrame,
    date_col: str,
) -> pd.DataFrame:
    """
    For each trade, find nearest event before and after the specified date.

    Uses merge_asof for efficient temporal joins.

    Args:
        trades_df: Trades with _symbol, _trade_date, _expiry_date (must be pre-sorted by [_symbol, date_col])
        events_df: Events with _symbol, _event_date, _event_type (must be pre-sorted by [_symbol, _event_date])
        date_col: Which date to check ("_trade_date" or "_expiry_date")

    Returns:
        DataFrame with added columns: nearest_event_before, nearest_event_after,
        days_to_nearest_event_before, days_to_nearest_event_after, nearest_event_type_before,
        nearest_event_type_after
    """
    out = trades_df.copy()
    out = out.sort_values(date_col, kind="mergesort").reset_index(drop=True)

    events_sorted = events_df.copy()
    events_sorted = events_sorted.sort_values(["_symbol", "_event_date"], kind="mergesort").reset_index(drop=True)

    # Prepare event dataframes for merging
    events_before = events_sorted[["_symbol", "_event_date", "_event_type"]].copy()
    events_before.columns = ["_symbol", "event_before", "type_before"]
    events_before = events_before.sort_values("event_before", kind="mergesort").reset_index(drop=True)

    events_after = events_sorted[["_symbol", "_event_date", "_event_type"]].copy()
    events_after.columns = ["_symbol", "event_after", "type_after"]
    events_after = events_after.sort_values("event_after", kind="mergesort").reset_index(drop=True)

    # Find nearest event BEFORE the date
    prev_events = pd.merge_asof(
        out,
        events_before,
        by="_symbol",
        left_on=date_col,
        right_on="event_before",
        direction="backward",
        allow_exact_matches=True,
        suffixes=("_old", ""),
    )

    # Find nearest event AFTER the date
    next_events = pd.merge_asof(
        out,
        events_after,
        by="_symbol",
        left_on=date_col,
        right_on="event_after",
        direction="forward",
        allow_exact_matches=True,
        suffixes=("_old", ""),
    )

    # Calculate distances
    out["event_before"] = prev_events["event_before"]
    out["type_before"] = prev_events["type_before"]
    out["days_before"] = (out[date_col] - out["event_before"]).dt.days

    out["event_after"] = next_events["event_after"]
    out["type_after"] = next_events["type_after"]
    out["days_after"] = (out["event_after"] - out[date_col]).dt.days

    return out


def apply_exclusion_rules(
    trades_df: pd.DataFrame,
    exclusion_windows: Dict[str, ExclusionWindow],
    phase: str,  # "trade" or "expiry"
) -> pd.DataFrame:
    """
    Mark trades for exclusion based on proximity to events.

    Args:
        trades_df: DataFrame with event_before, event_after, type_before, type_after, days_before, days_after
        exclusion_windows: Dict mapping event_type to ExclusionWindow
        phase: "trade" (check tradeTime proximity) or "expiry" (check expirationDate proximity)

    Returns:
        DataFrame with added columns: exclude_{phase}, exclude_{phase}_reason
    """
    df = trades_df.copy()
    df[f"exclude_{phase}"] = False
    df[f"exclude_{phase}_reason"] = ""

    for idx, row in df.iterrows():
        excluded = False
        reason = []

        # Check event BEFORE
        if pd.notna(row["event_before"]) and pd.notna(row["type_before"]):
            event_type = row["type_before"]
            if event_type in exclusion_windows:
                window = exclusion_windows[event_type]
                days_before = row["days_before"]

                if phase == "trade":
                    max_days_before = window.days_before_trade
                    if 0 <= days_before <= max_days_before:
                        excluded = True
                        reason.append(f"{event_type}_before_{days_before}d")
                else:  # expiry
                    max_days_before = window.days_before_expiry
                    if 0 <= days_before <= max_days_before:
                        excluded = True
                        reason.append(f"{event_type}_before_{days_before}d")

        # Check event AFTER
        if pd.notna(row["event_after"]) and pd.notna(row["type_after"]):
            event_type = row["type_after"]
            if event_type in exclusion_windows:
                window = exclusion_windows[event_type]
                days_after = row["days_after"]

                if phase == "trade":
                    max_days_after = window.days_after_trade
                    if 0 <= days_after <= max_days_after:
                        excluded = True
                        reason.append(f"{event_type}_after_{days_after}d")
                else:  # expiry
                    max_days_after = window.days_after_expiry
                    if 0 <= days_after <= max_days_after:
                        excluded = True
                        reason.append(f"{event_type}_after_{days_after}d")

        df.at[idx, f"exclude_{phase}"] = excluded
        df.at[idx, f"exclude_{phase}_reason"] = "; ".join(reason)

    return df


def generate_report(
    original_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    excluded_df: pd.DataFrame,
    exclusion_windows: Dict[str, ExclusionWindow],
) -> str:
    """Generate filtering statistics report"""
    report = []
    report.append("=" * 70)
    report.append("Corporate Events Trade Filtering Report")
    report.append("=" * 70)
    report.append("")

    # Overall stats
    report.append("Overall Statistics:")
    report.append(f"  Original trades:        {len(original_df):,}")
    report.append(f"  Trades kept:            {len(filtered_df):,} ({100*len(filtered_df)/len(original_df):.1f}%)")
    report.append(f"  Trades excluded:        {len(excluded_df):,} ({100*len(excluded_df)/len(original_df):.1f}%)")
    report.append("")

    # Exclusion reasons breakdown
    if len(excluded_df) > 0:
        report.append("Exclusion Reasons (by phase):")

        # Trade phase exclusions
        trade_excluded = excluded_df[excluded_df["exclude_trade"]]
        if len(trade_excluded) > 0:
            report.append(f"  Trade phase:            {len(trade_excluded):,} trades")
            reasons = trade_excluded["exclude_trade_reason"].value_counts()
            for reason, count in reasons.head(10).items():
                report.append(f"    - {reason:40s} {count:,}")

        # Expiry phase exclusions
        expiry_excluded = excluded_df[excluded_df["exclude_expiry"]]
        if len(expiry_excluded) > 0:
            report.append(f"  Expiry phase:           {len(expiry_excluded):,} trades")
            reasons = expiry_excluded["exclude_expiry_reason"].value_counts()
            for reason, count in reasons.head(10).items():
                report.append(f"    - {reason:40s} {count:,}")

        report.append("")

    # Event type breakdown
    report.append("Exclusions by Event Type:")
    for event_type, window in exclusion_windows.items():
        # Count trades excluded due to this event type
        trade_count = excluded_df["exclude_trade_reason"].str.contains(event_type, na=False).sum()
        expiry_count = excluded_df["exclude_expiry_reason"].str.contains(event_type, na=False).sum()
        total = len(excluded_df[(excluded_df["exclude_trade_reason"].str.contains(event_type, na=False)) |
                                (excluded_df["exclude_expiry_reason"].str.contains(event_type, na=False))])

        report.append(f"  {event_type}:")
        report.append(f"    Window: ±{window.days_before_trade}/{window.days_after_trade} days (trade), "
                     f"±{window.days_before_expiry}/{window.days_after_expiry} days (expiry)")
        report.append(f"    Excluded: {total:,} trades ({trade_count} trade phase, {expiry_count} expiry phase)")

    report.append("")
    report.append("=" * 70)

    return "\n".join(report)


def main():
    """Main filtering pipeline"""
    print("=" * 70)
    print("Corporate Events Trade Filtering (a03)")
    print("=" * 70)
    print()

    # Load configuration
    try:
        config = load_config()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nPlease create corp_action_config.yaml with exclusion_windows configuration.")
        sys.exit(1)

    # Validate config
    if not config.trades_csv:
        print("ERROR: trades_input_csv not specified in config")
        sys.exit(1)
    if not config.events_csv:
        print("ERROR: output_csv (events file) not specified in config")
        sys.exit(1)
    if not config.output_csv:
        print("ERROR: filtered_trades_csv not specified in config")
        sys.exit(1)
    if not config.exclusion_windows:
        print("ERROR: No exclusion_windows defined in config")
        sys.exit(1)

    print(f"Configuration:")
    print(f"  Trades input:  {config.trades_csv}")
    print(f"  Events input:  {config.events_csv}")
    print(f"  Output:        {config.output_csv}")
    print(f"  Event types:   {', '.join(config.exclusion_windows.keys())}")
    print()

    # Load data
    trades_df = load_trades(
        config.trades_csv,
        config.symbol_col,
        config.trade_date_col,
        config.expiry_col,
    )

    events_df = load_events(config.events_csv)
    print()

    # Phase 1: Check tradeTime proximity
    print("Phase 1: Checking trade date proximity to events...")
    trades_with_trade_events = find_nearest_events(trades_df, events_df, "_trade_date")
    trades_with_trade_events = apply_exclusion_rules(
        trades_with_trade_events,
        config.exclusion_windows,
        phase="trade",
    )
    trade_excluded_count = trades_with_trade_events["exclude_trade"].sum()
    print(f"  {trade_excluded_count:,} trades flagged for exclusion")
    print()

    # Phase 2: Check expirationDate proximity
    print("Phase 2: Checking expiry date proximity to events...")
    trades_with_all_events = find_nearest_events(trades_with_trade_events, events_df, "_expiry_date")
    trades_with_all_events = apply_exclusion_rules(
        trades_with_all_events,
        config.exclusion_windows,
        phase="expiry",
    )
    expiry_excluded_count = trades_with_all_events["exclude_expiry"].sum()
    print(f"  {expiry_excluded_count:,} trades flagged for exclusion")
    print()

    # Combine exclusions
    trades_with_all_events["excluded"] = (
        trades_with_all_events["exclude_trade"] | trades_with_all_events["exclude_expiry"]
    )

    # Split into kept vs excluded
    kept_df = trades_with_all_events[~trades_with_all_events["excluded"]].copy()
    excluded_df = trades_with_all_events[trades_with_all_events["excluded"]].copy()

    # Generate report
    report = generate_report(trades_df, kept_df, excluded_df, config.exclusion_windows)
    print(report)

    # Save output
    print("\nSaving filtered trades...")

    # Drop temporary columns before saving
    cols_to_drop = [c for c in kept_df.columns if c.startswith("_") or c.startswith("exclude")
                    or c in ["excluded", "event_before", "event_after", "type_before", "type_after",
                            "days_before", "days_after"]]
    kept_df = kept_df.drop(columns=cols_to_drop)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(config.output_csv) or ".", exist_ok=True)
    kept_df.to_csv(config.output_csv, index=False)
    print(f"  Saved {len(kept_df):,} filtered trades → {config.output_csv}")

    # Optionally save excluded trades
    if config.keep_filtered_trades and config.filtered_csv:
        excluded_df.to_csv(config.filtered_csv, index=False)
        print(f"  Saved {len(excluded_df):,} excluded trades → {config.filtered_csv}")

    # Save report
    report_path = config.output_csv.replace(".csv", "_filter_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved filtering report → {report_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
