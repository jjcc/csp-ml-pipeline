"""
tail_scoring.py — Shared library for tail-loss classifier operations.

Provides model loading, scoring, threshold selection, and metrics for the
tail classifier pipeline (b03_train_tail / b04_score_tail).

The tail classifier is a VETO LAYER that runs AFTER the 4-bin winner
classifier.  It identifies "toxic" trades — those the winner model predicted
as top-bin (bin-3) but that actually fall in the worst bin (bin-0).

High tail_proba means "likely a catastrophic loser — avoid this trade."

Model pack structure (joblib dict)
------------------------------------
  model              — fitted LightGBM classifier
  calibrator         — IsotonicRegression calibrator (applied after model.predict_proba)
  features           — list[str] feature names (base features + BIN_PROB_FEATS)
  medians            — dict[str, float] training medians for imputation
  oof_best_threshold — float, best-F1 threshold from OOF cross-validation
  oof_auc            — float
  oof_avg_precision  — float
  tail_rate          — float, fraction of training data labeled as tail (bin-0)
  contamination_rate — float, fraction of pred-bin3 trades that were actually bin-0
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Model pack dataclass
# ---------------------------------------------------------------------------

@dataclass
class TailModelPack:
    """Typed wrapper around the raw joblib dict."""
    model: object                          # LightGBM classifier
    features: list                         # feature names (base + bin probs)
    medians: dict                          # training medians for imputation
    oof_best_threshold: float              # best-F1 threshold from OOF CV
    calibrator: object = None             # IsotonicRegression (optional)
    oof_auc: float = float("nan")
    oof_avg_precision: float = float("nan")
    tail_rate: float = float("nan")        # fraction of training data that is tail
    contamination_rate: float = float("nan")  # pred-bin3 trades actually bin-0


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_tail_model(path: str) -> TailModelPack:
    """Load a saved tail model pack from *path* and return a TailModelPack."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Tail model pack not found: {path}")
    raw = joblib.load(path)
    required = {"model", "features", "medians", "oof_best_threshold"}
    missing = required - set(raw.keys())
    if missing:
        raise ValueError(f"Tail model pack is missing keys: {missing}")
    return TailModelPack(
        model=raw["model"],
        features=raw["features"],
        medians=raw["medians"],
        oof_best_threshold=raw["oof_best_threshold"],
        calibrator=raw.get("calibrator"),
        oof_auc=raw.get("oof_auc", float("nan")),
        oof_avg_precision=raw.get("oof_avg_precision", float("nan")),
        tail_rate=raw.get("tail_rate", float("nan")),
        contamination_rate=raw.get("contamination_rate", float("nan")),
    )


def save_tail_model(pack: TailModelPack, path: str) -> None:
    """Save a TailModelPack to *path* as a joblib file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump({
        "model":               pack.model,
        "calibrator":          pack.calibrator,
        "features":            pack.features,
        "medians":             pack.medians,
        "oof_best_threshold":  pack.oof_best_threshold,
        "oof_auc":             pack.oof_auc,
        "oof_avg_precision":   pack.oof_avg_precision,
        "tail_rate":           pack.tail_rate,
        "contamination_rate":  pack.contamination_rate,
    }, path)


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def fill_features(df: pd.DataFrame, feat_list: list,
                  medians: Optional[dict] = None) -> tuple[np.ndarray, dict]:
    """Fill features with medians (training) or compute medians from df (training).

    Args:
        df:        Input DataFrame (must contain or approximate feat_list columns)
        feat_list: Ordered list of feature names expected by the model
        medians:   If provided (scoring), use these training medians.
                   If None (training), compute medians from df.

    Returns:
        (X, medians_dict)  — X is float64 ndarray, medians_dict maps name→float
    """
    Xdf = df.copy()
    computed_medians = {}
    for col in feat_list:
        if col not in Xdf.columns:
            Xdf[col] = np.nan
        if col == "gex_missing":
            Xdf[col] = Xdf[col].fillna(1)
            computed_medians[col] = 0.0
        else:
            if medians is not None and col in medians:
                med = medians[col]
            else:
                med_val = Xdf[col].median(skipna=True)
                med = float(med_val) if pd.notna(med_val) else 0.0
            computed_medians[col] = med
            Xdf[col] = Xdf[col].fillna(med)
    return Xdf[feat_list].astype(float).values, computed_medians


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------

def build_tail_labels_from_bins(df: pd.DataFrame) -> np.ndarray:
    """Label tail trades using the 4-bin winner OOF output.

    A trade is labeled as tail (1) when y_true == 0 (i.e., the actual return
    fell in the worst quartile / bin-0 of the 4-bin winner model).

    Args:
        df: DataFrame containing a 'y_true' column (from winner_scores_oof.csv)

    Returns:
        y — int array of 0/1 tail labels
    """
    if "y_true" not in df.columns:
        raise ValueError(
            "'y_true' column not found. Ensure winner_scores_oof.csv (bins4 format) "
            "has been merged into df before calling build_tail_labels_from_bins()."
        )
    return (df["y_true"] == 0).astype(int).values


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_tail_data(df: pd.DataFrame, pack: TailModelPack,
                    proba_col: str = "tail_proba") -> tuple[pd.DataFrame, np.ndarray]:
    """Score df with the tail model.  Returns (out_df, proba_array).

    If pack.calibrator is present (IsotonicRegression), the raw model
    probabilities are calibrated before returning.
    """
    X, _ = fill_features(df, pack.features, medians=pack.medians)
    proba = pack.model.predict_proba(X)[:, 1]
    if pack.calibrator is not None:
        proba = pack.calibrator.transform(proba)
    out = df.copy()
    out[proba_col] = proba
    return out, proba


def add_bin_prob_features(df: pd.DataFrame, winner_model_pack: dict) -> pd.DataFrame:
    """Apply a 4-bin winner model to df and add p_bin0-3 + conflict_score columns.

    Called by b04_score_tail when scoring new (unlabeled) trades that don't
    already have winner OOF probability columns.

    Args:
        df:               Input trades DataFrame (must have all winner features)
        winner_model_pack: Dict loaded from winner model pkl (binary-mode pack
                          is rejected; must be a bins4 final-model pack that
                          supports predict_proba returning shape (N, 4))

    Returns:
        df copy with p_bin0, p_bin1, p_bin2, p_bin3, conflict_score added.
    """
    model   = winner_model_pack["model"]
    features = winner_model_pack["features"]
    medians  = winner_model_pack.get("medians", {})

    Xdf = df.copy()
    for col in features:
        if col not in Xdf.columns:
            Xdf[col] = np.nan
        med = float(medians.get(col, 0.0))
        Xdf[col] = pd.to_numeric(Xdf[col], errors="coerce").fillna(med)

    X = Xdf[features].astype(float).values
    proba = model.predict_proba(X)   # shape (N, 4) for bins4 model
    if proba.shape[1] != 4:
        raise ValueError(
            f"Winner model returned {proba.shape[1]} classes — expected 4. "
            "Ensure b01 was trained with WINNER_LABEL_MODE=bins4."
        )

    out = df.copy()
    out["p_bin0"] = proba[:, 0]
    out["p_bin1"] = proba[:, 1]
    out["p_bin2"] = proba[:, 2]
    out["p_bin3"] = proba[:, 3]
    out["conflict_score"] = proba[:, 0] * proba[:, 3]
    return out


def apply_tail_threshold(df: pd.DataFrame, proba_col: str,
                         pred_col: str, threshold: float) -> pd.DataFrame:
    """Add a binary prediction column based on threshold."""
    out = df.copy()
    out[pred_col] = (out[proba_col] >= threshold).astype(int)
    return out


def select_tail_threshold(proba: np.ndarray,
                          y: Optional[np.ndarray],
                          pack: TailModelPack,
                          fixed_threshold: Optional[float] = None,
                          target_precision: Optional[float] = None,
                          target_recall: Optional[float] = None) -> float:
    """Select the decision threshold.

    Priority:
      1. fixed_threshold (explicit override)
      2. target_precision / target_recall calibration on held-out data
      3. oof_best_threshold from the model pack
    """
    if fixed_threshold is not None:
        return float(fixed_threshold)

    if y is not None and len(np.unique(y)) > 1:
        prec_arr, rec_arr, thr_arr = precision_recall_curve(y, proba)
        # prec_arr and rec_arr have one extra element; thr_arr is one shorter
        if target_precision is not None:
            mask = prec_arr[:-1] >= target_precision
            if mask.any():
                return float(thr_arr[mask][-1])   # highest recall at that precision
        if target_recall is not None:
            mask = rec_arr[:-1] >= target_recall
            if mask.any():
                return float(thr_arr[mask][0])    # highest precision at that recall

    return float(pack.oof_best_threshold)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_tail_metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    """Compute classification metrics for a tail model."""
    if len(np.unique(y)) < 2:
        return {"auc_roc": float("nan"), "auc_prc": float("nan")}
    return {
        "auc_roc": float(roc_auc_score(y, proba)),
        "auc_prc": float(average_precision_score(y, proba)),
    }


def write_tail_metrics(out_dir: str, metrics: dict, extra: dict | None = None) -> None:
    """Write tail model metrics to a JSON file in out_dir."""
    combined = {**metrics, **(extra or {})}
    path = os.path.join(out_dir, "tail_classifier_metrics.json")
    with open(path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"[INFO] Metrics → {path}")
