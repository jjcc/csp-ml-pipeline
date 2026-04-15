#!/usr/bin/env python3
"""
production_data.py

Shared data processing functions for production/on-fly data preparation.
Contains functions extracted from task_score_tail_winner.py for reuse.
"""
import os
import pandas as pd
from datetime import datetime, time
from service.data_prepare import COMMON_START_DATE, _load_vix, add_macro_features
from service.get_vix import get_current_vix, init_driver, url_vix
from service.preprocess import merge_gex
from service.env_config import getenv


def parse_target_time(s: str) -> time:
    """Parse time string in HH:MM format."""
    try:
        hh, mm = s.split(":")
        return time(int(hh), int(mm))
    except Exception:
        return time(11, 0)


def add_features(target_minutes: int, option_file: str, target_date: str) -> pd.DataFrame:
    """Add GEX and macro features to option data.

    Args:
        target_minutes: Target time in minutes from midnight
        option_file: Path to option CSV file
        target_date: Date string in YYYY-MM-DD format

    Returns:
        DataFrame with added features
    """
    gex_base_dir = getenv("GEX_BASE_DIR")
    trades = pd.read_csv(option_file)
    df_o = merge_gex(trades, gex_base_dir, target_minutes=target_minutes)

    # Add macro features using shared function
    today = datetime.now()
    # Get VIX (handles both real-time and historic)
    vix_df = get_vix(today, target_date=datetime.strptime(target_date, "%Y-%m-%d").date())

    # Use shared macro features function with pre-built VIX DataFrame
    PX_BASE_DIR = getenv("MACRO_PX_BASE_DIR", "").strip()
    df_o = add_macro_features(df_o, vix_df, PX_BASE_DIR)
    return df_o


def get_vix(today: datetime, target_date=None) -> pd.DataFrame:
    """Get VIX data either from historical CSV or real-time fetch.

    Args:
        today: Current datetime
        target_date: Target date (if different from today, use historical data)

    Returns:
        DataFrame with columns ['trade_date', 'VIX']
    """
    if target_date is not None:
        if target_date != today.date():
            VIX_CSV = getenv("MACRO_VIX_CSV", "").strip() or None
            start_date = target_date - pd.Timedelta(days=1)
            end_date = target_date + pd.Timedelta(days=1)
            #TODO: use merge later
            exist_vix = pd.read_csv(VIX_CSV, parse_dates=["Date"])
            minimun_date = exist_vix["Date"].min()
            st = minimun_date
            st = pd.to_datetime(st)
            #st = pd.to_datetime(COMMON_START_DATE)
            vix = _load_vix(VIX_CSV, st, end_date)
            vix_df = pd.DataFrame({"trade_date": vix.index, "VIX": vix.values})
            print(f"Target date {target_date} is not today {today.date()}, skip VIX fetch.")
            return vix_df

    driver = init_driver(headless=True)
    vix_value = get_current_vix(url_vix, driver)
    try:
        vix_value = float(vix_value)
        today_date = today.date()
        today_date = pd.to_datetime(today_date)
    except ValueError:
        vix_value = 16.35
    vix_df = pd.DataFrame({"trade_date": [today_date], "VIX": [vix_value]})
    return vix_df
