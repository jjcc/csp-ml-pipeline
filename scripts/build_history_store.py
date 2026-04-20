#!/usr/bin/env python3
"""
Build the full CSP history store (a01 -> a04) for a date range.

This is a prep-only runner. It updates the large history datasets and stops
before any rolling-window training/scoring steps.
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name, "")
    value = value.strip() if isinstance(value, str) else ""
    return value if value else default


def _coalesce_str(cli_value: str | None, env_name: str, default: str) -> str:
    if cli_value is not None and str(cli_value).strip():
        return str(cli_value).strip()
    return _env_str(env_name, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the parquet-backed full history store with a01 -> a04. "
            "All options can be set in .env via HISTORY_BUILD_* variables."
        )
    )
    parser.add_argument("--start-date", default=None, help="Oldest usable history date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=None, help="Latest trade date to include in the prep pass (YYYY-MM-DD).")
    parser.add_argument("--config", default=None, help="Base config file to clone for the history build.")
    parser.add_argument("--python", default=None, help="Python executable to use for subprocess pipeline runs.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def update_dataset_dates(cfg: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp) -> None:
    cfg.setdefault("dataset", {})
    cfg["dataset"]["events_start_date"] = start.strftime("%Y-%m-%d")
    cfg["dataset"]["events_end_date"] = end.strftime("%Y-%m-%d")


def build_prep_config(
    base_cfg: dict[str, Any],
    overall_start: pd.Timestamp,
    overall_end: pd.Timestamp,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    update_dataset_dates(cfg, overall_start, overall_end)
    cutoff = overall_end + pd.Timedelta(days=21)
    cfg.setdefault("dataset", {})
    cfg["dataset"]["cutoff_date"] = cutoff.strftime("%Y-%m-%d")
    return cfg


def run_step(python_exe: str, script_rel: str, config_path: Path) -> None:
    env = os.environ.copy()
    env["CSP_CONFIG_PATH"] = str(config_path)
    cmd = [python_exe, str(PROJECT_ROOT / script_rel)]
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    args = parse_args()

    config_path = Path(
        _coalesce_str(args.config, "HISTORY_BUILD_CONFIG", str(DEFAULT_CONFIG_PATH))
    )
    python_exe = _coalesce_str(args.python, "HISTORY_BUILD_PYTHON", sys.executable)
    start_date = pd.Timestamp(
        _coalesce_str(args.start_date, "HISTORY_BUILD_START_DATE", "2025-04-28")
    ).normalize()
    end_date = pd.Timestamp(
        _coalesce_str(args.end_date, "HISTORY_BUILD_END_DATE", pd.Timestamp.today().strftime("%Y-%m-%d"))
    ).normalize()

    if end_date < start_date:
        raise SystemExit("HISTORY_BUILD_END_DATE must be on or after HISTORY_BUILD_START_DATE.")

    base_cfg = load_yaml(config_path)
    prep_cfg = build_prep_config(base_cfg=base_cfg, overall_start=start_date, overall_end=end_date)

    config_dir = PROJECT_ROOT / prep_cfg.get("common", {}).get("output_dir", "output") / "backfill" / "configs"
    prep_cfg_path = config_dir / f"history_store_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.yaml"
    save_yaml(prep_cfg_path, prep_cfg)

    print(
        f"[INFO] Full-history store build: "
        f"{prep_cfg['dataset']['events_start_date']} -> {prep_cfg['dataset']['events_end_date']}"
    )
    print(f"[INFO] Temp config: {prep_cfg_path}")

    for script in (
        "pipeline/a01_build_features.py",
        "pipeline/a02_collect_events.py",
        "pipeline/a03_filter_trades.py",
        "pipeline/a04_label_data.py",
    ):
        print(f"[INFO] Running {script} ...")
        run_step(python_exe, script, prep_cfg_path)

    print("[SUCCESS] History store build complete.")


if __name__ == "__main__":
    main()
