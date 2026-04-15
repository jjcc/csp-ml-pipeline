#!/usr/bin/env python3
"""
b04_score_tail.py — Score trade candidates with the Tail-Loss Classifier.

Loads a trained tail model pack, preprocesses input data exactly as training,
applies probability scoring, selects a decision threshold, and writes scored
output CSV files.

A trade flagged by the tail classifier (is_tail_pred=1) is predicted to fall
in the worst-K% of outcomes — treat these as "avoid" signals.

Usage:
    python pipeline/b04_score_tail.py

Typical workflow (run after b03_train_tail.py):
    python pipeline/b04_score_tail.py
"""

import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service.env_config import getenv
from service.preprocess import add_dte_and_normalized_returns
from service.tail_scoring import (
    load_tail_model,
    score_tail_data,
    apply_tail_threshold,
    select_tail_threshold,
    build_tail_labels,
    calculate_tail_metrics,
    write_tail_metrics,
)
from service.utils import ensure_dir, prep_tail_training_df


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TailScoringConfig:
    csv_in:          str
    model_in:        str
    csv_out_dir:     str
    csv_out:         str
    proba_col:       str
    pred_col:        str
    label_on:        str
    tail_pct:        float
    fixed_threshold: Optional[float]
    target_precision: Optional[float]
    target_recall:   Optional[float]


def load_scoring_config() -> TailScoringConfig:
    """Load tail scoring configuration from config.yaml / .env."""
    csv_in   = getenv("TAILSCORING_SCORE_INPUT", "./candidates.csv")
    model_in = getenv("TAILSCORING_MODEL_IN", "")
    if not model_in:
        raise SystemExit(
            "TAILSCORING_MODEL_IN is not set.  "
            "Ensure tail_scoring.model_in is configured in config.yaml."
        )

    csv_out_dir = getenv("TAILSCORING_SCORE_OUT_FOLDER", "output/tails_score/default")
    csv_out     = os.path.join(csv_out_dir,
                               getenv("TAILSCORING_SCORE_OUT", "tail_scores.csv"))

    fixed_thr_str = getenv("TAILSCORING_THRESHOLD", "").strip()
    fixed_thr = float(fixed_thr_str) if fixed_thr_str else None

    prec_str = getenv("TAILSCORING_TARGET_PRECISION", "").strip()
    rec_str  = getenv("TAILSCORING_TARGET_RECALL",    "").strip()

    return TailScoringConfig(
        csv_in=csv_in,
        model_in=model_in,
        csv_out_dir=csv_out_dir,
        csv_out=csv_out,
        proba_col=getenv("TAILSCORING_PROBA_COL",  "tail_proba"),
        pred_col=getenv("TAILSCORING_PRED_COL",    "is_tail_pred"),
        label_on=getenv("TAIL_LABEL_ON",            "return_mon").strip(),
        tail_pct=float(getenv("TAIL_PCT",           "0.05")),
        fixed_threshold=fixed_thr,
        target_precision=float(prec_str) if prec_str else None,
        target_recall=float(rec_str) if rec_str else None,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_preprocess(cfg: TailScoringConfig) -> pd.DataFrame:
    """Load and preprocess score data — must match training transforms exactly."""
    df = pd.read_csv(cfg.csv_in)

    required = ["symbol", "tradeTime"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}.  "
            f"Available: {list(df.columns)}"
        )

    df = add_dte_and_normalized_returns(df)
    df = prep_tail_training_df(df)

    if "tradeTime" in df.columns:
        df["tradeTime"] = pd.to_datetime(df["tradeTime"], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_outputs(cfg: TailScoringConfig, out: pd.DataFrame,
                  chosen_thr: float, y: Optional[np.ndarray],
                  proba: np.ndarray) -> None:
    os.makedirs(cfg.csv_out_dir, exist_ok=True)
    ensure_dir(cfg.csv_out)
    out.to_csv(cfg.csv_out, index=False)

    # Metrics if labels are available
    if y is not None and len(np.unique(y)) > 1:
        metrics = calculate_tail_metrics(y, proba)
        write_tail_metrics(cfg.csv_out_dir, metrics)
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}  "
              f"AUC-PRC: {metrics['auc_prc']:.4f}")

    # Summary JSON
    n_flagged = int(out[cfg.pred_col].sum()) if cfg.pred_col in out.columns else None
    summary = {
        "rows":           int(len(out)),
        "threshold":      float(chosen_thr),
        "tails_flagged":  n_flagged,
        "tail_rate":      round(float(out[cfg.pred_col].mean()), 4) if n_flagged is not None else None,
        "safe_fraction":  round(float(1.0 - out[cfg.pred_col].mean()), 4) if n_flagged is not None else None,
    }
    json_path = Path(cfg.csv_out).with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Scores → {cfg.csv_out}")
    print(f"  Threshold={chosen_thr:.6f} | flagged={n_flagged} "
          f"({summary['tail_rate']*100:.1f}% of trades) | "
          f"safe={summary['safe_fraction']*100:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg  = load_scoring_config()
    pack = load_tail_model(cfg.model_in)
    print(f"[INFO] Tail model loaded from {cfg.model_in}")
    print(f"       tail_pct={pack.tail_pct:.3f}  label_on={pack.label_on}  "
          f"oof_auc={pack.oof_auc:.4f}")

    df = load_and_preprocess(cfg)
    print(f"[INFO] {len(df):,} rows loaded from {cfg.csv_in}")

    # Score
    out, proba = score_tail_data(df, pack, cfg.proba_col)

    # Labels (if available, for evaluation)
    y = None
    if cfg.label_on in df.columns:
        try:
            y, _ = build_tail_labels(df, cfg.label_on, pack.tail_pct)
            out["is_tail"] = y
        except Exception:
            pass

    # Threshold selection
    chosen_thr = select_tail_threshold(
        proba, y, pack,
        fixed_threshold=cfg.fixed_threshold,
        target_precision=cfg.target_precision,
        target_recall=cfg.target_recall,
    )

    # Apply threshold
    out = apply_tail_threshold(out, cfg.proba_col, cfg.pred_col, chosen_thr)

    # Write outputs
    write_outputs(cfg, out, chosen_thr, y, proba)

    print(f"\n✅  Tail scoring complete. {len(out):,} rows scored.")


if __name__ == "__main__":
    main()
