#!/usr/bin/env python3
"""
Run b03 (train tail) + b04 (score tail) for every available historical window.

Each run writes to its own dated output directory, so runs don't clobber each other.
Uses CSP_CONFIG_PATH to point each subprocess at a temporary config with the right dates.

Usage:
    python scripts/run_all_tail_runs.py
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

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Run configurations
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    date_tag:              str            # YYYYMMDD
    rolling_window_weeks:  int
    events_start_date:     str            # YYYY-MM-DD
    events_end_date:       Optional[str]  # None = skip b04
    cutoff_date:           str            # YYYY-MM-DD


RUNS = [
    RunConfig("20250804", 14, "2025-08-04", None,         "2025-08-14"),  # no score window
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
    """Run pipeline/script.py as subprocess; returns (returncode, combined output)."""
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


def extract_b03_metrics(run: RunConfig) -> dict:
    metrics_path = (
        REPO_ROOT / "output" / "tails_train"
        / f"v9_roll{run.rolling_window_weeks}w_{run.date_tag}"
        / "tail_classifier_metrics.json"
    )
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


def extract_b04_metrics(run: RunConfig) -> dict:
    score_path = (
        REPO_ROOT / "output" / "tails_score"
        / f"v9_roll{run.rolling_window_weeks}w_{run.date_tag}"
        / f"tail_scores_{run.date_tag}.csv"
    )
    if not score_path.exists():
        return {}
    scored = pd.read_csv(score_path)
    result: dict = {"rows": len(scored)}
    if "tail_proba" in scored.columns:
        result["proba_min"]  = round(float(scored["tail_proba"].min()), 4)
        result["proba_max"]  = round(float(scored["tail_proba"].max()), 4)
        result["proba_mean"] = round(float(scored["tail_proba"].mean()), 4)
    if "is_tail_pred" in scored.columns and "return_mon" in scored.columns:
        tail_cut = float(scored["return_mon"].quantile(0.05))
        scored["is_tail"] = (scored["return_mon"] <= tail_cut).astype(int)
        flagged  = scored["is_tail_pred"] == 1
        tp       = int(((flagged) & (scored["is_tail"] == 1)).sum())
        fp       = int(((flagged) & (scored["is_tail"] == 0)).sum())
        fn       = int(((~flagged) & (scored["is_tail"] == 1)).sum())
        result["actual_tails"]  = int(scored["is_tail"].sum())
        result["flagged"]       = int(flagged.sum())
        result["tp"]            = tp
        result["fp"]            = fp
        result["fn"]            = fn
        result["precision"]     = round(tp / max(tp + fp, 1), 4)
        result["recall"]        = round(tp / max(tp + fn, 1), 4)
        result["lift"]          = round(result["precision"] / 0.05, 2)
    return result


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    base_cfg = load_base_config()
    summary_rows = []

    for run in RUNS:
        print(f"\n{'='*60}")
        print(f"RUN: {run.date_tag}  window={run.rolling_window_weeks}w  "
              f"start={run.events_start_date}  end={run.events_end_date or 'N/A'}")
        print('='*60)

        # Verify required inputs exist
        merged_csv = REPO_ROOT / "output" / "data_merged" / \
            f"merged_roll{run.rolling_window_weeks}w_{run.date_tag}.csv"
        oof_csv = REPO_ROOT / "output" / "winner_train" / \
            f"v9_roll{run.rolling_window_weeks}w_{run.date_tag}" / "winner_scores_oof.csv"

        if not merged_csv.exists():
            print(f"  [SKIP] merged CSV not found: {merged_csv}")
            continue
        if not oof_csv.exists():
            print(f"  [SKIP] winner OOF not found: {oof_csv}")
            continue

        # Write temp config
        run_cfg_dict = make_run_config(base_cfg, run)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, dir=str(REPO_ROOT)
        ) as tmp:
            yaml.dump(run_cfg_dict, tmp)
            tmp_config_path = tmp.name

        env = {**os.environ, "CSP_CONFIG_PATH": tmp_config_path}

        try:
            # ── b03 train ────────────────────────────────────────────────
            rc_b03, _ = run_script("b03_train_tail.py", env, f"b03 {run.date_tag}")

            # ── b04 score (only if scoring window is defined) ────────────
            rc_b04 = None
            if run.events_end_date and rc_b03 == 0:
                rc_b04, _ = run_script("b04_score_tail.py", env, f"b04 {run.date_tag}")

        finally:
            os.unlink(tmp_config_path)

        # Collect metrics
        b03m = extract_b03_metrics(run)
        b04m = extract_b04_metrics(run)

        row = {
            "date_tag":     run.date_tag,
            "window_weeks": run.rolling_window_weeks,
            "start":        run.events_start_date,
            "end":          run.events_end_date or "—",
            "b03_status":   "OK" if rc_b03 == 0 else "FAIL",
            "oof_auc":      round(b03m.get("auc_roc", float("nan")), 4),
            "oof_prc":      round(b03m.get("auc_prc", float("nan")), 4),
            "tail_rate":    b03m.get("tail_rate", ""),
            "b04_status":   "OK" if rc_b04 == 0 else ("SKIP" if rc_b04 is None else "FAIL"),
            "scored_rows":  b04m.get("rows", ""),
            "actual_tails": b04m.get("actual_tails", ""),
            "flagged":      b04m.get("flagged", ""),
            "precision":    b04m.get("precision", ""),
            "recall":       b04m.get("recall", ""),
            "lift":         b04m.get("lift", ""),
        }
        summary_rows.append(row)
        print(f"\n  b03 OOF AUC={row['oof_auc']}  PR-AUC={row['oof_prc']}")
        if b04m:
            print(f"  b04 flagged={row['flagged']}/{row['scored_rows']}  "
                  f"prec={row['precision']:.1%}  recall={row['recall']:.1%}  "
                  f"lift={row['lift']}x")

    # ── Print summary table ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    df = pd.DataFrame(summary_rows)
    print(df.to_string(index=False))

    # ── Save summary CSV ─────────────────────────────────────────────────
    out_path = REPO_ROOT / "output" / "tail_runs_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSummary → {out_path}")


if __name__ == "__main__":
    main()
