#!/usr/bin/env python3
"""
Stock split detection using yfinance split history.

Two modes:
1. Direct API: Use yfinance Ticker.splits (requires API call per symbol)
2. Cached detection: Analyze Close/Adj Close ratios from existing price cache (experimental)

Mode 1 is recommended as it's authoritative. Mode 2 is kept for reference but may not work
due to yfinance retroactively adjusting Close prices.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import yfinance as yf


def detect_splits_from_prices(
    price_df: pd.DataFrame,
    min_ratio_change: float = 0.01,
    min_split_size: float = 1.5
) -> List[Dict]:
    """
    Detect stock splits from Close vs Adj Close price differences.

    Logic:
    - Adj Close accounts for splits, dividends, etc.
    - Close is the raw market price
    - When Close/Adj Close ratio changes significantly = split happened
    - The ratio change magnitude tells us the split factor

    Args:
        price_df: DataFrame with 'Close' and 'Adj Close' columns, index is dates
        min_ratio_change: Minimum ratio change to consider (default 1% = 0.01)
        min_split_size: Minimum split factor to report (default 1.5x = 3:2 split)

    Returns:
        List of dicts with keys: date, split_factor, close, adj_close, ratio_before, ratio_after

    Example:
        For a 2:1 split on 2024-06-07:
        - Before: Close=1000, Adj Close=500, ratio=2.0
        - After:  Close=500,  Adj Close=500, ratio=1.0
        - Detection: ratio changed from 2.0 -> 1.0, split_factor = 2.0
    """
    if price_df is None or len(price_df) == 0:
        return []

    if 'Close' not in price_df.columns or 'Adj Close' not in price_df.columns:
        return []

    df = price_df.copy()
    df = df.sort_index()

    # Calculate Close/Adj Close ratio
    df['split_ratio'] = df['Close'] / df['Adj Close']

    # Remove inf/nan
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['split_ratio'])

    if len(df) < 2:
        return []

    # Detect ratio changes (shifts indicate splits)
    df['ratio_change'] = df['split_ratio'].diff().abs()

    # Find days where ratio changed significantly
    split_candidates = df[df['ratio_change'] > min_ratio_change].copy()

    splits = []
    for idx in split_candidates.index:
        try:
            current_ratio = df.loc[idx, 'split_ratio']
            prev_idx = df.index[df.index < idx][-1] if any(df.index < idx) else None

            if prev_idx is None:
                continue

            prev_ratio = df.loc[prev_idx, 'split_ratio']

            # Calculate split factor
            # If ratio went from 2.0 -> 1.0, it's a 2:1 split
            # If ratio went from 3.0 -> 1.0, it's a 3:1 split
            split_factor = prev_ratio / current_ratio if current_ratio != 0 else 1.0

            # Only report meaningful splits (>= 1.5x)
            if split_factor >= min_split_size or split_factor <= 1/min_split_size:
                splits.append({
                    'date': idx.strftime('%Y-%m-%d'),
                    'split_factor': round(split_factor, 4),
                    'close': df.loc[idx, 'Close'],
                    'adj_close': df.loc[idx, 'Adj Close'],
                    'ratio_before': round(prev_ratio, 4),
                    'ratio_after': round(current_ratio, 4),
                    'ratio_change': round(df.loc[idx, 'ratio_change'], 4),
                })
        except Exception as e:
            print(f"[WARN] Split detection error at {idx}: {e}")
            continue

    return splits


def fetch_splits_yfinance(
    symbols: List[str],
    date_range: Optional[Tuple[str, str]] = None,
    sleep_seconds: float = 0.1
) -> pd.DataFrame:
    """
    Fetch stock split history directly from yfinance API.

    This is the RECOMMENDED method as it uses yfinance's authoritative split data.

    Args:
        symbols: List of ticker symbols
        date_range: Optional (start_date, end_date) tuple in 'YYYY-MM-DD' format
        sleep_seconds: Delay between API calls to be polite to yfinance

    Returns:
        DataFrame with columns: symbol, date, split_factor, split_ratio
    """
    all_splits = []

    start_dt = pd.to_datetime(date_range[0]) if date_range else None
    end_dt = pd.to_datetime(date_range[1]) if date_range else None

    for i, symbol in enumerate(symbols):
        try:
            # Handle ticker format (BRK.B stays as BRK.B for yfinance)
            ticker = yf.Ticker(symbol)
            splits = ticker.splits

            if splits.empty:
                continue

            # Filter by date range if provided
            if start_dt or end_dt:
                mask = pd.Series([True] * len(splits), index=splits.index)
                if start_dt:
                    mask &= (splits.index >= start_dt)
                if end_dt:
                    mask &= (splits.index <= end_dt)
                splits = splits[mask]

            for split_date, split_factor in splits.items():
                all_splits.append({
                    'symbol': symbol,
                    'date': split_date.strftime('%Y-%m-%d'),
                    'split_factor': float(split_factor),
                    'split_ratio': format_split_ratio(float(split_factor)),
                })

            # Be polite to yfinance
            if sleep_seconds > 0 and i < len(symbols) - 1:
                time.sleep(sleep_seconds)

        except Exception as e:
            print(f"[WARN] Failed to fetch splits for {symbol}: {e}")
            continue

    if not all_splits:
        return pd.DataFrame(columns=['symbol', 'date', 'split_factor', 'split_ratio'])

    df = pd.DataFrame(all_splits)
    df = df.sort_values(['symbol', 'date'])
    return df


def detect_splits_for_symbols(
    symbols: List[str],
    cache_dir: str,
    date_range: Optional[Tuple[str, str]] = None,
    min_ratio_change: float = 0.1,
    min_split_size: float = 1.5
) -> pd.DataFrame:
    """
    Scan cached price files for all symbols and detect splits from Close/Adj Close ratios.

    NOTE: This is EXPERIMENTAL and may not work reliably because yfinance retroactively
    adjusts Close prices. Use fetch_splits_yfinance() instead for authoritative data.

    Args:
        symbols: List of ticker symbols to check
        cache_dir: Directory containing cached parquet files (e.g., 'output/price_cache')
        date_range: Optional (start_date, end_date) tuple in 'YYYY-MM-DD' format
        min_ratio_change: Minimum ratio change threshold (default 0.1 = 10%)
        min_split_size: Minimum split factor to report

    Returns:
        DataFrame with columns: symbol, date, split_factor, close, adj_close, ratio_before, ratio_after
    """
    all_splits = []

    for symbol in symbols:
        # Handle special ticker conversions (BRK.B -> BRK-B for yfinance)
        cache_symbol = symbol.replace('.', '-')
        cache_path = os.path.join(cache_dir, f"{cache_symbol}.parquet")

        if not os.path.exists(cache_path):
            continue

        try:
            price_df = pd.read_parquet(cache_path)
            price_df.index = pd.to_datetime(price_df.index)

            # Filter by date range if provided
            if date_range:
                start, end = date_range
                mask = (price_df.index >= start) & (price_df.index <= end)
                price_df = price_df[mask]

            splits = detect_splits_from_prices(
                price_df,
                min_ratio_change=min_ratio_change,
                min_split_size=min_split_size
            )

            for split in splits:
                all_splits.append({
                    'symbol': symbol,
                    **split
                })

        except Exception as e:
            print(f"[WARN] Failed to process {symbol}: {e}")
            continue

    if not all_splits:
        return pd.DataFrame(columns=['symbol', 'date', 'split_factor'])

    df = pd.DataFrame(all_splits)
    df = df.sort_values(['symbol', 'date'])
    return df


def format_split_ratio(split_factor: float) -> str:
    """
    Convert split factor to human-readable ratio.

    Examples:
        2.0 -> "2:1"
        3.0 -> "3:1"
        0.5 -> "1:2" (reverse split)
        1.5 -> "3:2"
    """
    if split_factor >= 1:
        # Forward split (e.g., 2:1, 3:1)
        if split_factor == int(split_factor):
            return f"{int(split_factor)}:1"
        else:
            # Handle fractional splits (e.g., 3:2 = 1.5)
            # Try common ratios
            if abs(split_factor - 1.5) < 0.01:
                return "3:2"
            elif abs(split_factor - 2.5) < 0.01:
                return "5:2"
            else:
                return f"{split_factor:.2f}:1"
    else:
        # Reverse split (e.g., 1:2, 1:5)
        inv = 1 / split_factor
        if inv == int(inv):
            return f"1:{int(inv)}"
        else:
            return f"1:{inv:.2f}"


# CLI for testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python split_detector.py <symbol1> [symbol2] ...")
        print("Example: python split_detector.py NVDA TSLA AAPL")
        print("\nFetches split history from yfinance API")
        sys.exit(1)

    symbols = sys.argv[1:]

    print(f"Fetching splits for {len(symbols)} symbols from yfinance...")
    df = fetch_splits_yfinance(symbols)

    if len(df) == 0:
        print("No splits detected.")
    else:
        print(f"\nDetected {len(df)} split(s):\n")
        for _, row in df.iterrows():
            print(f"{row['symbol']:6s} {row['date']} - {row['split_ratio']:8s} split (factor={row['split_factor']:.2f})")
