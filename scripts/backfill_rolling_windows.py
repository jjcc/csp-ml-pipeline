#!/usr/bin/env python3
"""
Run the CSP pipeline across consecutive date-stamped rolling score windows.

Workflow
--------
1. Run the full-history prep pass once (a01 -> a04) against a temporary config
   spanning the requested backfill date range.
2. Split labeled trade dates into consecutive score windows.
3. For each window:
   - write a date-stamped score input CSV for that window
   - generate a temporary config with that window's dates
   - run a05 -> b04 against that temporary config
   - collect per-window metrics into a summary CSV

This keeps the user's main config.yaml untouched while still using the existing
date-stamped output conventions throughout the pipeline.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class ScoreWindow:
    start: pd.Timestamp
    end: pd.Timestamp
    trade_dates: tuple[pd.Timestamp, ...]

    @property
    def tag(self) -> str:
        return self.start.strftime("%Y%m%d")


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name, "")
    value = value.strip() if isinstance(value, str) else ""
    return value if value else default


def _env_int(name: str, default: int = 0) -> int:
    raw = _env_str(name, "")
    return int(raw) if raw else default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env_str(name, "")
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _coalesce_str(cli_value: str | None, env_name: str, default: str) -> str:
    if cli_value is not None and str(cli_value).strip():
        return str(cli_value).strip()
    return _env_str(env_name, default)


def _coalesce_int(cli_value: int | None, env_name: str, default: int) -> int:
    if cli_value is not None:
        return int(cli_value)
    return _env_int(env_name, default)


def _coalesce_bool(cli_value: bool | None, env_name: str, default: bool) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    return _env_bool(env_name, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill the rolling-window CSP pipeline with date-stamped outputs. "
                    "All options can be set in .env via BACKFILL_* variables."
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Oldest usable history date (YYYY-MM-DD). This is not the first "
             "score window date; the first score window starts after the rolling "
             "warm-up period.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Latest trade date to include when building score windows (YYYY-MM-DD). "
             "Defaults to the latest labeled trade date.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Base config file to clone for prep and per-window runs.",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Python executable to use for subprocess pipeline runs.",
    )
    parser.add_argument(
        "--score-window-trade-days",
        type=int,
        default=None,
        help="Number of trading days per score window. Defaults to the current "
             "config's business-day span between dataset.events_start_date and "
             "dataset.events_end_date.",
    )
    parser.add_argument(
        "--rolling-weeks",
        type=int,
        default=None,
        help="Override rolling_window_weeks for all temporary configs. "
             "Defaults to the value in the base config.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Limit the number of score windows to run. 0 = all windows.",
    )
    parser.add_argument(
        "--skip-prep",
        action="store_true",
        default=None,
        help="Skip the full-history prep pass and reuse the existing labeled CSV.",
    )
    parser.add_argument(
        "--skip-tail",
        action="store_true",
        default=None,
        help="Run only a05 -> b02 and skip b03/b04.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=None,
        help="Skip windows whose date-stamped outputs already exist.",
    )
    parser.add_argument(
        "--drop-partial-last-window",
        action="store_true",
        default=None,
        help="Drop the final score window if it has fewer trade days than the configured chunk size.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def resolve_labeled_csv(base_cfg: dict[str, Any]) -> Path:
    output_dir = base_cfg.get("common", {}).get("output_dir", "output")
    output_csv = base_cfg.get("dataset", {}).get("output_csv", "")
    if not output_csv:
        raise SystemExit("dataset.output_csv must be set in the base config.")
    return PROJECT_ROOT / output_dir / "data_labeled" / output_csv


def infer_score_window_trade_days(base_cfg: dict[str, Any]) -> int:
    dataset_cfg = base_cfg.get("dataset", {})
    start = pd.Timestamp(dataset_cfg.get("events_start_date"))
    end = pd.Timestamp(dataset_cfg.get("events_end_date"))
    trade_days = len(pd.bdate_range(start, end))
    if trade_days <= 0:
        raise SystemExit(
            "Could not infer score-window length from dataset.events_start_date "
            "and dataset.events_end_date."
        )
    return int(trade_days)


def load_labeled_data(labeled_csv: Path) -> pd.DataFrame:
    if not labeled_csv.is_file():
        raise SystemExit(
            f"Labeled CSV not found: {labeled_csv}\n"
            "Run the prep pass or remove --skip-prep."
        )
    df = pd.read_csv(labeled_csv)
    if "tradeTime" not in df.columns:
        raise SystemExit(f"'tradeTime' column not found in labeled CSV: {labeled_csv}")
    df["tradeTime"] = pd.to_datetime(df["tradeTime"], errors="coerce")
    df = df[df["tradeTime"].notna()].copy()
    df["trade_date"] = df["tradeTime"].dt.normalize()
    return df


def build_score_windows(
    trade_dates: list[pd.Timestamp],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp | None,
    window_trade_days: int,
    drop_partial_last_window: bool,
    max_windows: int,
) -> list[ScoreWindow]:
    eligible = [
        d for d in trade_dates
        if d >= start_date and (end_date is None or d <= end_date)
    ]
    windows: list[ScoreWindow] = []
    for i in range(0, len(eligible), window_trade_days):
        chunk = eligible[i:i + window_trade_days]
        if not chunk:
            continue
        if len(chunk) < window_trade_days and drop_partial_last_window:
            break
        windows.append(ScoreWindow(start=chunk[0], end=chunk[-1], trade_dates=tuple(chunk)))
        if max_windows and len(windows) >= max_windows:
            break
    return windows


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


def build_window_config(
    base_cfg: dict[str, Any],
    window: ScoreWindow,
    score_input_rel: str,
    rolling_weeks_override: int,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    update_dataset_dates(cfg, window.start, window.end)
    if rolling_weeks_override:
        cfg["rolling_window_weeks"] = int(rolling_weeks_override)
    cfg.setdefault("winnerscore", {})
    cfg["winnerscore"]["score_input"] = score_input_rel
    cfg.setdefault("tailscoring", {})
    cfg["tailscoring"]["score_input"] = score_input_rel
    return cfg


def relative_to_project(path: Path) -> str:
    return os.path.relpath(path, PROJECT_ROOT)


def run_step(python_exe: str, script_rel: str, config_path: Path) -> None:
    env = os.environ.copy()
    env["CSP_CONFIG_PATH"] = str(config_path)
    cmd = [python_exe, str(PROJECT_ROOT / script_rel)]
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def score_window_csv_path(backfill_dir: Path, window: ScoreWindow) -> Path:
    name = f"score_{window.tag}_{window.end.strftime('%Y%m%d')}.csv"
    return backfill_dir / "score_windows" / name


def write_score_window_csv(df: pd.DataFrame, window: ScoreWindow, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = (df["trade_date"] >= window.start) & (df["trade_date"] <= window.end)
    score_df = df.loc[mask].copy()
    score_df.drop(columns=["trade_date"], errors="ignore", inplace=True)
    score_df.to_csv(path, index=False)
    return int(len(score_df))


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def window_output_paths(output_dir: str, rolling_weeks: int, tag: str) -> dict[str, Path]:
    base = PROJECT_ROOT / output_dir
    return {
        "winner_train_dir": base / "winner_train" / f"v9_roll{rolling_weeks}w_{tag}",
        "winner_score_dir": base / "winner_score" / f"v9_roll{rolling_weeks}w_{tag}",
        "tail_train_dir": base / "tails_train" / f"v9_roll{rolling_weeks}w_{tag}",
        "tail_score_dir": base / "tails_score" / f"v9_roll{rolling_weeks}w_{tag}",
        "merged_csv": base / "data_merged" / f"merged_roll{rolling_weeks}w_{tag}.csv",
    }

def collect_summary_row(
    output_dir: str,
    rolling_weeks: int,
    window: ScoreWindow,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    score_csv: Path,
    score_rows: int,
    train_rows: int,
    status: str,
    previous_window: ScoreWindow | None,
) -> dict[str, Any]:
    tag = window.tag
    paths = window_output_paths(output_dir, rolling_weeks, tag)
    winner_train = load_json_if_exists(paths["winner_train_dir"] / "winner_classifier_metrics.json")
    winner_score = load_json_if_exists(paths["winner_score_dir"] / f"scores_{tag}.json")
    tail_train = load_json_if_exists(paths["tail_train_dir"] / "tail_classifier_metrics.json")
    tail_score = load_json_if_exists(paths["tail_score_dir"] / f"tail_scores_{tag}.json")

    row: dict[str, Any] = {
        "train_start": train_start.strftime("%Y-%m-%d"),
        "train_end": train_end.strftime("%Y-%m-%d"),
        "window_start": window.start.strftime("%Y-%m-%d"),
        "window_end": window.end.strftime("%Y-%m-%d"),
        "window_tag": tag,
        "train_history_weeks": rolling_weeks,
        "trade_days_in_window": len(window.trade_dates),
        "score_rows": score_rows,
        "train_rows": train_rows,
        "status": status,
        "score_input_csv": relative_to_project(score_csv),
        "previous_window_tag": previous_window.tag if previous_window is not None else "",
        "previous_window_score_included": previous_window is not None,
    }

    def merge_prefixed(prefix: str, data: dict[str, Any]) -> None:
        for key, value in data.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[f"{prefix}{key}"] = value

    merge_prefixed("winner_train_", winner_train)
    merge_prefixed("winner_score_", winner_score)
    merge_prefixed("tail_train_", tail_train)
    merge_prefixed("tail_score_", tail_score)
    return row


def persist_summary(rows: list[dict[str, Any]], summary_csv: Path) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_csv, index=False)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    args = parse_args()

    base_config_path = Path(
        _coalesce_str(args.config, "BACKFILL_CONFIG", str(DEFAULT_CONFIG_PATH))
    ).resolve()
    base_cfg = load_yaml(base_config_path)
    output_dir = base_cfg.get("common", {}).get("output_dir", "output")
    rolling_weeks = _coalesce_int(
        args.rolling_weeks,
        "BACKFILL_ROLLING_WEEKS",
        int(base_cfg.get("rolling_window_weeks", 14)),
    )
    score_window_trade_days = int(
        _coalesce_int(
            args.score_window_trade_days,
            "BACKFILL_SCORE_WINDOW_TRADE_DAYS",
            infer_score_window_trade_days(base_cfg),
        )
    )

    history_start_date = pd.Timestamp(
        _coalesce_str(args.start_date, "BACKFILL_START_DATE", "2025-04-25")
    ).normalize()
    end_date_raw = _coalesce_str(args.end_date, "BACKFILL_END_DATE", "")
    end_date = pd.Timestamp(end_date_raw).normalize() if end_date_raw else None
    python_exe = _coalesce_str(args.python, "BACKFILL_PYTHON", sys.executable)
    max_windows = _coalesce_int(args.max_windows, "BACKFILL_MAX_WINDOWS", 0)
    skip_prep = _coalesce_bool(args.skip_prep, "BACKFILL_SKIP_PREP", False)
    skip_tail = _coalesce_bool(args.skip_tail, "BACKFILL_SKIP_TAIL", False)
    skip_existing = _coalesce_bool(args.skip_existing, "BACKFILL_SKIP_EXISTING", False)
    drop_partial_last_window = _coalesce_bool(
        args.drop_partial_last_window,
        "BACKFILL_DROP_PARTIAL_LAST_WINDOW",
        False,
    )

    backfill_dir = PROJECT_ROOT / output_dir / "backfill"
    config_dir = backfill_dir / "configs"
    summary_csv = backfill_dir / "rolling_window_summary.csv"
    first_score_candidate = history_start_date + timedelta(weeks=rolling_weeks)

    if not skip_prep:
        prep_cfg = build_prep_config(
            base_cfg=base_cfg,
            overall_start=history_start_date,
            overall_end=end_date if end_date is not None else history_start_date,
        )
        # If no explicit end date was provided, infer it from available option filenames by
        # reading the latest labeled date after prep.  For the initial prep pass, widen to
        # the latest configured option snapshot date by using the latest GEX/option date we
        # can see from the mounted folders later in the process.
        if end_date is None:
            option_glob = sorted((PROJECT_ROOT / "option" / "put").glob("coveredPut_*.csv"))
            if not option_glob:
                raise SystemExit("No option snapshot files found under option/put.")
            last_name = option_glob[-1].name
            inferred_end = pd.Timestamp(last_name.split("_")[1]).normalize()
            prep_cfg["dataset"]["events_end_date"] = inferred_end.strftime("%Y-%m-%d")
            prep_cfg["dataset"]["cutoff_date"] = (inferred_end + pd.Timedelta(days=21)).strftime("%Y-%m-%d")

        prep_cfg_path = config_dir / "prep_full_history.yaml"
        save_yaml(prep_cfg_path, prep_cfg)

        print(
            f"[INFO] Full-history prep pass: "
            f"{prep_cfg['dataset']['events_start_date']} -> {prep_cfg['dataset']['events_end_date']}"
        )
        for script in (
            "pipeline/a01_build_features.py",
            "pipeline/a02_collect_events.py",
            "pipeline/a03_filter_trades.py",
            "pipeline/a04_label_data.py",
        ):
            print(f"[INFO] Running {script} ...")
            run_step(python_exe, script, prep_cfg_path)

    labeled_csv = resolve_labeled_csv(base_cfg)
    labeled_df = load_labeled_data(labeled_csv)
    if end_date is None:
        end_date = pd.Timestamp(labeled_df["trade_date"].max()).normalize()

    trade_dates = sorted(pd.Series(labeled_df["trade_date"].unique()).tolist())
    windows = build_score_windows(
        trade_dates=trade_dates,
        start_date=first_score_candidate,
        end_date=end_date,
        window_trade_days=score_window_trade_days,
        drop_partial_last_window=drop_partial_last_window,
        max_windows=max_windows,
    )
    if not windows:
        raise SystemExit("No score windows were generated. Check the date range and labeled CSV.")

    print(
        f"[INFO] History start={history_start_date.date()} | "
        f"first score candidate={first_score_candidate.date()} | "
        f"generated {len(windows)} score windows "
        f"({score_window_trade_days} trade days each, rolling_weeks={rolling_weeks})."
    )

    summary_rows: list[dict[str, Any]] = []
    for idx, window in enumerate(windows, start=1):
        tag = window.tag
        score_csv = score_window_csv_path(backfill_dir, window)
        score_rows = write_score_window_csv(labeled_df, window, score_csv)

        train_start = window.start - timedelta(weeks=rolling_weeks)
        train_end = window.start - pd.Timedelta(days=1)
        train_mask = (labeled_df["trade_date"] >= train_start) & (labeled_df["trade_date"] < window.start)
        train_rows = int(train_mask.sum())
        previous_window = windows[idx - 2] if idx > 1 else None

        print(
            f"[INFO] Window {idx}/{len(windows)} "
            f"{window.start.date()} -> {window.end.date()} "
            f"| train_rows={train_rows:,} score_rows={score_rows:,}"
        )

        status = "completed"
        if score_rows == 0:
            status = "skipped_no_score_rows"
        elif train_rows == 0:
            status = "skipped_no_training_rows"

        window_cfg = build_window_config(
            base_cfg=base_cfg,
            window=window,
            score_input_rel=relative_to_project(score_csv),
            rolling_weeks_override=rolling_weeks,
        )
        window_cfg_path = config_dir / f"window_{tag}.yaml"
        save_yaml(window_cfg_path, window_cfg)

        output_paths = window_output_paths(output_dir, rolling_weeks, tag)
        if skip_existing and status == "completed":
            winner_json = output_paths["winner_score_dir"] / f"scores_{tag}.json"
            tail_json = output_paths["tail_score_dir"] / f"tail_scores_{tag}.json"
            already_done = winner_json.is_file() and (skip_tail or tail_json.is_file())
            if already_done:
                status = "skipped_existing"

        if status == "completed":
            try:
                for script in (
                    "pipeline/a05_merge_datasets.py",
                    "pipeline/b01_train_winner.py",
                    "pipeline/b02_score_winner.py",
                ):
                    print(f"[INFO] Running {script} for {tag} ...")
                    run_step(python_exe, script, window_cfg_path)

                if not skip_tail:
                    for script in (
                        "pipeline/b03_train_tail.py",
                        "pipeline/b04_score_tail.py",
                    ):
                        print(f"[INFO] Running {script} for {tag} ...")
                        run_step(python_exe, script, window_cfg_path)
            except subprocess.CalledProcessError as exc:
                status = f"failed_{Path(exc.cmd[-1]).name}"
                print(f"[ERROR] Window {tag} failed while running {exc.cmd[-1]}")

        summary_rows.append(
            collect_summary_row(
                output_dir=output_dir,
                rolling_weeks=rolling_weeks,
                window=window,
                train_start=train_start,
                train_end=train_end,
                score_csv=score_csv,
                score_rows=score_rows,
                train_rows=train_rows,
                status=status,
                previous_window=previous_window,
            )
        )
        persist_summary(summary_rows, summary_csv)

    print(f"\n[INFO] Backfill summary written to {summary_csv}")


if __name__ == "__main__":
    main()
