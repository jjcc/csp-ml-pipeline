#!/usr/bin/env python3
"""
Run b01 (train winner) + b02 (score) for every available historical window,
then immediately re-run b03 (train tail) + b04 (score) so the tail classifier
reflects the updated winner OOF probabilities and GEX features.

Writes output/winner_runs_summary.csv with per-run metrics.

Usage:
    python scripts/run_all_winner_runs.py
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Run configurations  (same table as run_all_tail_runs.py)
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    date_tag:             str
    rolling_window_weeks: int
    events_start_date:    str
    events_end_date:      Optional[str]   # None = skip scoring
    cutoff_date:          str


RUNS = [
    RunConfig("20250804", 14, "2025-08-04", None,         "2025-08-14"),
    RunConfig("20250818", 16, "2025-08-18", "2025-09-05", "2025-09-15"),
    RunConfig("20250901", 16, "2025-09-01", "2025-09-12", "2025-09-22"),
    RunConfig("20250908", 16, "2025-09-08", "2025-09-08", "2025-09-15"),
    RunConfig("20250915", 16, "2025-09-15", "2025-09-15", "2025-09-22"),
    RunConfig("20250929", 16, "2025-09-29", "2025-09-29", "2025-10-06"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_base_config() -> dict:
    with open(REPO_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def make_run_config(base: dict, run: RunConfig) -> dict:
    cfg = copy.deepcopy(base)
    cfg["rolling_window_weeks"] = run.rolling_window_weeks
    cfg["dataset"]["events_start_date"] = run.events_start_date
    cfg["dataset"]["events_end_date"]   = run.events_end_date or run.events_start_date
    cfg["dataset"]["cutoff_date"]       = run.cutoff_date
    return cfg


def run_script(script: str, env: dict, label: str) -> tuple[int, str]:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "pipeline" / script)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    out = result.stdout + result.stderr
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"  [{label}] {status}")
    for line in out.splitlines():
        print(f"    {line}")
    return result.returncode, out


def extract_winner_train_metrics(run: RunConfig) -> dict:
    path = (
        REPO_ROOT / "output" / "winner_train"
        / f"v9_roll{run.rolling_window_weeks}w_{run.date_tag}"
        / "winner_classifier_metrics.json"
    )
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def extract_winner_score_metrics(run: RunConfig) -> dict:
    score_path = (
        REPO_ROOT / "output" / "winner_score"
        / f"v9_roll{run.rolling_window_weeks}w_{run.date_tag}"
        / f"scores_{run.date_tag}.csv"
    )
    if not score_path.exists():
        return {}
    df = pd.read_csv(score_path)
    result: dict = {"rows": len(df)}
    if "win_proba" in df.columns:
        result["proba_min"]  = round(float(df["win_proba"].min()), 4)
        result["proba_max"]  = round(float(df["win_proba"].max()), 4)
    if "win_predict" in df.columns and "return_mon" in df.columns:
        pred = df["win_predict"] == 1
        labeled_win = (df["return_mon"] > 0.02).astype(int)
        tp = int(((pred) & (labeled_win == 1)).sum())
        fp = int(((pred) & (labeled_win == 0)).sum())
        fn = int(((~pred) & (labeled_win == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        coverage  = float(pred.mean())
        # Spread: mean return of top-20% vs bottom-20% by win_proba
        top20 = df.nlargest(max(1, int(len(df) * 0.2)), "win_proba")["return_mon"].mean()
        bot20 = df.nsmallest(max(1, int(len(df) * 0.2)), "win_proba")["return_mon"].mean()
        result["coverage"]   = round(coverage, 4)
        result["precision"]  = round(precision, 4)
        result["recall"]     = round(recall, 4)
        result["top20_ret"]  = round(top20, 3)
        result["bot20_ret"]  = round(bot20, 3)
        result["spread"]     = round(top20 - bot20, 3)
    return result


def extract_tail_metrics(run: RunConfig) -> dict:
    metrics_path = (
        REPO_ROOT / "output" / "tails_train"
        / f"v9_roll{run.rolling_window_weeks}w_{run.date_tag}"
        / "tail_classifier_metrics.json"
    )
    if not metrics_path.exists():
        return {}
    with open(metrics_path) as f:
        return json.load(f)


def extract_tail_score_metrics(run: RunConfig) -> dict:
    score_path = (
        REPO_ROOT / "output" / "tails_score"
        / f"v9_roll{run.rolling_window_weeks}w_{run.date_tag}"
        / f"tail_scores_{run.date_tag}.csv"
    )
    if not score_path.exists():
        return {}
    df = pd.read_csv(score_path)
    result: dict = {"rows": len(df)}
    if "is_tail_pred" in df.columns and "return_mon" in df.columns:
        tail_cut = float(df["return_mon"].quantile(0.05))
        df["is_tail"] = (df["return_mon"] <= tail_cut).astype(int)
        flagged = df["is_tail_pred"] == 1
        tp = int(((flagged) & (df["is_tail"] == 1)).sum())
        fp = int(((flagged) & (df["is_tail"] == 0)).sum())
        fn = int(((~flagged) & (df["is_tail"] == 1)).sum())
        result["tail_flagged"]    = int(flagged.sum())
        result["tail_precision"]  = round(tp / max(tp + fp, 1), 4)
        result["tail_recall"]     = round(tp / max(tp + fn, 1), 4)
        result["tail_lift"]       = round(result["tail_precision"] / 0.05, 2)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    base_cfg = load_base_config()
    summary_rows = []

    for run in RUNS:
        print(f"\n{'='*65}")
        print(f"RUN: {run.date_tag}  window={run.rolling_window_weeks}w  "
              f"{run.events_start_date} → {run.events_end_date or 'no scoring'}")
        print('='*65)

        # Verify required inputs
        merged_csv = REPO_ROOT / "output" / "data_merged" / \
            f"merged_roll{run.rolling_window_weeks}w_{run.date_tag}.csv"
        if not merged_csv.exists():
            print(f"  [SKIP] merged CSV not found: {merged_csv}")
            continue

        run_cfg_dict = make_run_config(base_cfg, run)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, dir=str(REPO_ROOT)
        ) as tmp:
            yaml.dump(run_cfg_dict, tmp)
            tmp_path = tmp.name

        env = {**os.environ, "CSP_CONFIG_PATH": tmp_path}

        try:
            # ── b01: train winner (with GEX features) ──────────────────────────
            rc_b01, _ = run_script("b01_train_winner.py", env, f"b01 {run.date_tag}")

            # ── b02: score winner ──────────────────────────────────────────────
            rc_b02 = None
            if run.events_end_date and rc_b01 == 0:
                rc_b02, _ = run_script("b02_score_winner.py", env, f"b02 {run.date_tag}")

            # ── b03: re-train tail with updated winner OOF ─────────────────────
            rc_b03 = None
            if rc_b01 == 0:
                rc_b03, _ = run_script("b03_train_tail.py", env, f"b03 {run.date_tag}")

            # ── b04: score tail ────────────────────────────────────────────────
            rc_b04 = None
            if run.events_end_date and rc_b03 == 0:
                rc_b04, _ = run_script("b04_score_tail.py", env, f"b04 {run.date_tag}")

        finally:
            os.unlink(tmp_path)

        # Collect metrics
        wt = extract_winner_train_metrics(run)
        ws = extract_winner_score_metrics(run)
        tt = extract_tail_metrics(run)
        ts = extract_tail_score_metrics(run)

        n_feats = wt.get("n_features", "")
        spread  = wt.get("top_minus_bottom_spread", float("nan"))
        acc     = wt.get("accuracy", float("nan"))

        row = {
            "date_tag":       run.date_tag,
            "window_w":       run.rolling_window_weeks,
            "start":          run.events_start_date,
            "end":            run.events_end_date or "—",
            "b01":            "OK" if rc_b01 == 0 else "FAIL",
            "n_feats":        n_feats,
            "accuracy":       round(acc, 4) if acc == acc else "nan",
            "spread":         round(spread, 3) if spread == spread else "nan",
            "b02":            "OK" if rc_b02 == 0 else ("SKIP" if rc_b02 is None else "FAIL"),
            "w_rows":         ws.get("rows", ""),
            "w_coverage":     ws.get("coverage", ""),
            "w_precision":    ws.get("precision", ""),
            "w_spread":       ws.get("spread", ""),
            "b03":            "OK" if rc_b03 == 0 else ("SKIP" if rc_b03 is None else "FAIL"),
            "tail_auc":       round(tt.get("auc_roc", float("nan")), 4),
            "b04":            "OK" if rc_b04 == 0 else ("SKIP" if rc_b04 is None else "FAIL"),
            "t_flagged":      ts.get("tail_flagged", ""),
            "t_precision":    ts.get("tail_precision", ""),
            "t_lift":         ts.get("tail_lift", ""),
        }
        summary_rows.append(row)

        print(f"\n  b01 accuracy={row['accuracy']}  spread={row['spread']}  n_feats={n_feats}")
        if ws:
            print(f"  b02 rows={ws.get('rows','')}  coverage={ws.get('coverage','')}  "
                  f"precision={ws.get('precision','')}  spread={ws.get('spread','')}")
        if tt:
            print(f"  b03 tail_auc={row['tail_auc']}")
        if ts:
            print(f"  b04 flagged={ts.get('tail_flagged','')}  "
                  f"prec={ts.get('tail_precision','')}  lift={ts.get('tail_lift','')}x")

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SUMMARY")
    print('='*65)
    df = pd.DataFrame(summary_rows)
    print(df.to_string(index=False))

    out_path = REPO_ROOT / "output" / "winner_runs_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSummary → {out_path}")


if __name__ == "__main__":
    main()
