#!/usr/bin/env python3
"""
b14_score_tail_gex.py — Score candidates with the GEX-Only Tail Classifier.

Standalone — no winner model required.

Pipeline:
  1. Load new GEX indicators from NEW_GEX_FEATURE_FOLDER / tailgexscoring.gex_folder
  2. Load candidate trades CSV
  3. Backward asof merge: GEX.capture_dt <= trade.tradeTime, per symbol
  4. Load b13 model pack → encode regime → score → apply threshold
  5. Write scored CSV + summary JSON

High tail_gex_proba => "GEX regime suggests catastrophic-loss risk."

Usage:
    python pipeline/b14_score_tail_gex.py

Required config (config.yaml / .env):
    tailgexscoring.model_in          — path to b13 model pack (.pkl)
    tailgexscoring.score_input       — CSV of candidate trades
    tailgexscoring.score_out_folder  — output directory
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
from service.tail_scoring import apply_tail_threshold, fill_features
from service.utils import ensure_dir
from service.table_store import read_table, table_exists

# reuse GEX loading helpers from b13
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b13_train_tail_gex import load_gex_indicators, merge_gex_to_trades


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GexScoringConfig:
    csv_in:           str
    model_in:         str
    csv_out_dir:      str
    csv_out:          str
    proba_col:        str
    pred_col:         str
    fixed_threshold:  Optional[float]
    target_precision: Optional[float]
    target_recall:    Optional[float]
    gex_folder:       str


def load_scoring_config() -> GexScoringConfig:
    csv_in   = getenv("TAILGEXSCORING_SCORE_INPUT", "./candidates.csv")
    model_in = getenv("TAILGEXSCORING_MODEL_IN", "")
    if not model_in:
        raise SystemExit(
            "TAILGEXSCORING_MODEL_IN not set.  "
            "Ensure tailgexscoring.model_in is set in config.yaml."
        )

    csv_out_dir = getenv("TAILGEXSCORING_SCORE_OUT_FOLDER", "output/tails_gex_score/default")
    csv_out     = os.path.join(csv_out_dir, getenv("TAILGEXSCORING_SCORE_OUT", "tail_gex_scores.csv"))

    fixed_thr_str = getenv("TAILGEXSCORING_THRESHOLD", "").strip()
    prec_str      = getenv("TAILGEXSCORING_TARGET_PRECISION", "").strip()
    rec_str       = getenv("TAILGEXSCORING_TARGET_RECALL", "").strip()

    gex_folder = getenv("TAILGEXSCORING_GEX_FOLDER", "").strip()
    if not gex_folder:
        gex_folder = getenv("NEW_GEX_FEATURE_FOLDER", "").strip()
    if not gex_folder:
        raise SystemExit(
            "GEX folder not configured.  Set NEW_GEX_FEATURE_FOLDER in .env "
            "or tailgexscoring.gex_folder in config.yaml."
        )

    return GexScoringConfig(
        csv_in=csv_in,
        model_in=model_in,
        csv_out_dir=csv_out_dir,
        csv_out=csv_out,
        proba_col=getenv("TAILGEXSCORING_PROBA_COL", "tail_gex_proba"),
        pred_col=getenv("TAILGEXSCORING_PRED_COL", "is_tail_gex_pred"),
        fixed_threshold=float(fixed_thr_str) if fixed_thr_str else None,
        target_precision=float(prec_str) if prec_str else None,
        target_recall=float(rec_str) if rec_str else None,
        gex_folder=gex_folder,
    )


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def select_threshold(pack: dict, fixed: Optional[float],
                     target_prec: Optional[float],
                     target_rec: Optional[float],
                     proba: np.ndarray,
                     y: Optional[np.ndarray] = None) -> float:
    if fixed is not None:
        return float(fixed)
    # No labels at scoring time — fall back to pack's best-F1 threshold
    return float(pack.get("oof_best_threshold", 0.5))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_outputs(cfg: GexScoringConfig, out: pd.DataFrame, threshold: float) -> None:
    os.makedirs(cfg.csv_out_dir, exist_ok=True)
    out.to_csv(cfg.csv_out, index=False)

    n_flagged = int(out[cfg.pred_col].sum()) if cfg.pred_col in out.columns else None
    tail_rate = float(out[cfg.pred_col].mean()) if n_flagged is not None else None

    summary = {
        "rows":          int(len(out)),
        "threshold":     float(threshold),
        "tails_flagged": n_flagged,
        "tail_rate":     round(tail_rate, 4) if tail_rate is not None else None,
        "safe_fraction": round(1.0 - tail_rate, 4) if tail_rate is not None else None,
    }
    json_path = Path(cfg.csv_out).with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    safe_pct = (1.0 - tail_rate) * 100 if tail_rate is not None else float("nan")
    print(f"  Scores → {cfg.csv_out}")
    print(f"  Threshold={threshold:.6f} | flagged={n_flagged} "
          f"({tail_rate*100:.1f}% of trades) | safe={safe_pct:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_scoring_config()

    # ── Load model pack ───────────────────────────────────────────────────────
    print(f"[INFO] Loading GEX tail model: {cfg.model_in}")
    if not os.path.isfile(cfg.model_in):
        raise FileNotFoundError(f"Model not found: {cfg.model_in}")
    pack = joblib.load(cfg.model_in)
    print(f"       oof_auc={pack.get('oof_auc', float('nan')):.4f}  "
          f"tail_rate={pack.get('tail_rate', float('nan')):.1%}  "
          f"threshold={pack.get('oof_best_threshold', 0.5):.4f}")

    # ── Load GEX indicators ───────────────────────────────────────────────────
    gex = load_gex_indicators(cfg.gex_folder)

    # ── Load and merge candidates ─────────────────────────────────────────────
    from service.table_store import resolve_read_path
    actual_path = resolve_read_path(cfg.csv_in) if table_exists(cfg.csv_in) else cfg.csv_in
    print(f"[INFO] Loading candidates: {actual_path}")
    df = read_table(cfg.csv_in) if table_exists(cfg.csv_in) else pd.read_csv(cfg.csv_in)
    if "baseSymbol" not in df.columns and "symbol" in df.columns:
        df = df.rename(columns={"symbol": "baseSymbol"})
    print(f"[INFO] {len(df):,} rows loaded")

    # Filter to active scoring window before GEX merge (cheaper on fewer rows).
    df["tradeTime"] = pd.to_datetime(df["tradeTime"], errors="coerce")
    score_start = getenv("DATASET_EVENTS_START_DATE", "").strip()
    score_end   = getenv("DATASET_EVENTS_END_DATE",   "").strip()
    if score_start:
        n_before = len(df)
        df = df[df["tradeTime"] >= pd.Timestamp(score_start)]
        if score_end:
            df = df[df["tradeTime"] <= pd.Timestamp(score_end)]
        print(f"[INFO] Scoring window filter {score_start} → {score_end or 'open'}: "
              f"{n_before:,} → {len(df):,} rows")

    df = merge_gex_to_trades(df, gex)

    # ── Score ─────────────────────────────────────────────────────────────────
    feat_list = pack["features"]
    medians   = pack["medians"]
    X, _      = fill_features(df, feat_list, medians=medians)

    proba = pack["model"].predict_proba(X)[:, 1]
    if pack.get("calibrator") is not None:
        proba = pack["calibrator"].transform(proba)

    out = df.copy()
    out[cfg.proba_col] = proba

    # ── Threshold ─────────────────────────────────────────────────────────────
    threshold = select_threshold(pack, cfg.fixed_threshold,
                                 cfg.target_precision, cfg.target_recall, proba)

    out = apply_tail_threshold(out, cfg.proba_col, cfg.pred_col, threshold)

    # ── Write outputs ─────────────────────────────────────────────────────────
    write_outputs(cfg, out, threshold)
    print(f"\n✅  GEX tail scoring complete. {len(out):,} rows scored.")


if __name__ == "__main__":
    main()
