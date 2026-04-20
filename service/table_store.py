from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


_TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def parquet_path(path: str) -> str:
    """Return the canonical parquet path for a configured dataset path."""
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return str(p)
    if p.suffix:
        return str(p.with_suffix(".parquet"))
    return str(Path(f"{path}.parquet"))


def csv_path(path: str) -> str:
    """Return the CSV export path matching a configured dataset path."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return str(p)
    if p.suffix:
        return str(p.with_suffix(".csv"))
    return str(Path(f"{path}.csv"))


def table_exists(path: str) -> bool:
    return os.path.isfile(parquet_path(path)) or os.path.isfile(csv_path(path))


def resolve_read_path(path: str) -> str:
    parquet = parquet_path(path)
    if os.path.isfile(parquet):
        return parquet

    csv_export = csv_path(path)
    if os.path.isfile(csv_export):
        return csv_export

    return parquet


def read_table(path: str, columns: list[str] | None = None, **kwargs) -> pd.DataFrame:
    """Read a parquet-backed history table, falling back to CSV exports."""
    read_path = resolve_read_path(path)
    index_col = kwargs.pop("index_col", None)

    if read_path.endswith(".parquet"):
        df = pd.read_parquet(read_path, columns=columns)
        if index_col is not None and index_col in df.columns:
            df = df.set_index(index_col)
        return df

    if columns is not None and "usecols" not in kwargs:
        kwargs["usecols"] = columns
    return pd.read_csv(read_path, **kwargs)


def should_write_csv_export(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return explicit

    load_dotenv(override=False)
    return os.getenv("HISTORY_WRITE_CSV_EXPORTS", "0").strip().lower() in _TRUE_VALUES


def write_table(
    df: pd.DataFrame,
    path: str,
    *,
    append: bool = False,
    index: bool = False,
    write_csv_export: bool | None = None,
) -> str:
    """Write the canonical parquet table and optionally refresh a CSV export."""
    parquet = parquet_path(path)
    Path(parquet).parent.mkdir(parents=True, exist_ok=True)

    if append and table_exists(path):
        existing = read_table(path)
        df = pd.concat([existing, df], ignore_index=False)

    df.to_parquet(parquet, index=index)

    if should_write_csv_export(write_csv_export):
        df.to_csv(csv_path(path), index=index)

    return parquet
