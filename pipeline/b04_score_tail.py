#!/usr/bin/env python3
"""
b04_score_tail.py — Score trade candidates with the Tail-Loss Classifier.

The tail classifier is a VETO LAYER.  It identifies trades that are likely to
produce catastrophic losses (bin-0 outcomes) even when the winner model ranked
them highly.

Because the tail classifier uses 4-bin winner probabilities (p_bin0-3,
conflict_score) as features, this script applies the winner model first to
generate those features, then runs the tail model.

Pipeline:
  1. Load winner model → apply to new trades → add p_bin0-3, conflict_score
  2. Load tail model pack → score → apply isotonic calibrator → apply threshold
  3. Write scored CSV + summary JSON

High tail_proba => "likely a catastrophic loser — do not enter this trade."

Usage:
    python pipeline/b04_score_tail.py

Required config (config.yaml / .env):
    tailscoring.model_in          — path to tail model pack (.pkl from b03)
    tailscoring.winner_model_in   — path to winner model pack (.pkl from b01)
    tailscoring.score_input       — CSV of candidate trades to score
    tailscoring.score_out_folder  — output directory
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service.env_config import getenv
from service.preprocess import add_dte_and_normalized_returns
from service.tail_scoring import (
    add_bin_prob_features,
    apply_tail_threshold,
    load_tail_model,
    score_tail_data,
    select_tail_threshold,
    write_tail_metrics,
)
from service.utils import ensure_dir


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TailScoringConfig:
    csv_in:           str
    model_in:         str
    winner_model_in:  str           # winner model for computing p_bin0-3 features
    csv_out_dir:      str
    csv_out:          str
    proba_col:        str
    pred_col:         str
    fixed_threshold:  Optional[float]
    target_precision: Optional[float]
    target_recall:    Optional[float]


def load_scoring_config() -> TailScoringConfig:
    """Load tail scoring configuration from config.yaml / .env."""
    csv_in = getenv("TAILSCORING_SCORE_INPUT", "./candidates.csv")

    model_in = getenv("TAILSCORING_MODEL_IN", "")
    if not model_in:
        raise SystemExit(
            "TAILSCORING_MODEL_IN is not set.  "
            "Ensure tail_scoring.model_in is configured in config.yaml."
        )

    winner_model_in = getenv("TAILSCORING_WINNER_MODEL_IN", "")
    if not winner_model_in:
        raise SystemExit(
            "TAILSCORING_WINNER_MODEL_IN is not set.\n"
            "The tail classifier requires the 4-bin winner model to compute "
            "p_bin0-3 features before scoring.\n"
            "Set tailscoring.winner_model_in in config.yaml to the winner model .pkl."
        )

    csv_out_dir = getenv("TAILSCORING_SCORE_OUT_FOLDER", "output/tails_score/default")
    csv_out     = os.path.join(
        csv_out_dir, getenv("TAILSCORING_SCORE_OUT", "tail_scores.csv")
    )

    fixed_thr_str = getenv("TAILSCORING_THRESHOLD", "").strip()
    fixed_thr     = float(fixed_thr_str) if fixed_thr_str else None

    prec_str = getenv("TAILSCORING_TARGET_PRECISION", "").strip()
    rec_str  = getenv("TAILSCORING_TARGET_RECALL",    "").strip()

    return TailScoringConfig(
        csv_in=csv_in,
        model_in=model_in,
        winner_model_in=winner_model_in,
        csv_out_dir=csv_out_dir,
        csv_out=csv_out,
        proba_col=getenv("TAILSCORING_PROBA_COL", "tail_proba"),
        pred_col=getenv("TAILSCORING_PRED_COL",   "is_tail_pred"),
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

    # Accept either column name for the symbol
    if "baseSymbol" not in df.columns and "symbol" in df.columns:
        df = df.rename(columns={"symbol": "baseSymbol"})

    required = ["baseSymbol", "tradeTime"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}.  "
            f"Available: {list(df.columns)}"
        )

    df = add_dte_and_normalized_returns(df)

    if "tradeTime" in df.columns:
        df["tradeTime"] = pd.to_datetime(df["tradeTime"], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_outputs(cfg: TailScoringConfig, out: pd.DataFrame,
                  chosen_thr: float) -> None:
    os.makedirs(cfg.csv_out_dir, exist_ok=True)
    out.to_csv(cfg.csv_out, index=False)

    n_flagged  = int(out[cfg.pred_col].sum()) if cfg.pred_col in out.columns else None
    tail_rate  = float(out[cfg.pred_col].mean()) if n_flagged is not None else None

    summary = {
        "rows":          int(len(out)),
        "threshold":     float(chosen_thr),
        "tails_flagged": n_flagged,
        "tail_rate":     round(tail_rate, 4) if tail_rate is not None else None,
        "safe_fraction": round(1.0 - tail_rate, 4) if tail_rate is not None else None,
    }
    json_path = Path(cfg.csv_out).with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    safe_pct = (1.0 - tail_rate) * 100 if tail_rate is not None else float("nan")
    print(f"  Scores → {cfg.csv_out}")
    print(f"  Threshold={chosen_thr:.6f} | flagged={n_flagged} "
          f"({tail_rate*100:.1f}% of trades) | safe={safe_pct:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_scoring_config()

    # ── Load winner model (needed for p_bin0-3 feature generation) ───────────
    print(f"[INFO] Loading winner model: {cfg.winner_model_in}")
    if not os.path.isfile(cfg.winner_model_in):
        raise FileNotFoundError(f"Winner model not found: {cfg.winner_model_in}")
    winner_pack = joblib.load(cfg.winner_model_in)

    # ── Load tail model ───────────────────────────────────────────────────────
    print(f"[INFO] Loading tail model: {cfg.model_in}")
    pack = load_tail_model(cfg.model_in)
    print(f"       oof_auc={pack.oof_auc:.4f}  "
          f"contamination_rate={pack.contamination_rate:.1%}  "
          f"tail_rate={pack.tail_rate:.1%}  "
          f"threshold={pack.oof_best_threshold:.4f}")

    # ── Load and preprocess candidate trades ──────────────────────────────────
    df = load_and_preprocess(cfg)
    print(f"[INFO] {len(df):,} rows loaded from {cfg.csv_in}")

    # ── Apply winner model to generate p_bin0-3 and conflict_score features ──
    print(f"[INFO] Applying winner model to generate bin probability features…")
    df = add_bin_prob_features(df, winner_pack)
    print(f"       p_bin3 mean={df['p_bin3'].mean():.3f}  "
          f"conflict_score mean={df['conflict_score'].mean():.4f}")

    # ── Score with tail model (applies calibrator if present) ─────────────────
    out, proba = score_tail_data(df, pack, cfg.proba_col)

    # ── Threshold selection (no labels available for new data) ────────────────
    chosen_thr = select_tail_threshold(
        proba, None, pack,
        fixed_threshold=cfg.fixed_threshold,
        target_precision=cfg.target_precision,
        target_recall=cfg.target_recall,
    )

    # ── Apply threshold ───────────────────────────────────────────────────────
    out = apply_tail_threshold(out, cfg.proba_col, cfg.pred_col, chosen_thr)

    # ── Write outputs ─────────────────────────────────────────────────────────
    write_outputs(cfg, out, chosen_thr)

    print(f"\n✅  Tail scoring complete. {len(out):,} rows scored.")


if __name__ == "__main__":
    main()
