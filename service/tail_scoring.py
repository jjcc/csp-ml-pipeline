"""
tail_scoring.py — Shared library for tail-loss classifier operations.

Provides model loading, scoring, threshold selection, and metrics for the
tail classifier pipeline (b03_train_tail / b04_score_tail).

The tail classifier identifies the worst-K% of CSP trades by monthly return —
contracts that are likely to produce catastrophic losses.  High tail_proba
means "likely a fat-tail loser"; filter these out before trading.

Model pack structure (joblib dict)
------------------------------------
  model              — fitted classifier (GradientBoosting or LightGBM)
  features           — list[str] feature names used at training time
  medians            — dict[str, float] training medians for imputation
  tail_pct           — float, fraction used to define tail (e.g. 0.05)
  label_on           — str, return column used for labeling (e.g. "return_mon")
  tail_cut_value     — float, the quantile threshold that was applied
  oof_best_threshold — float, best-F1 threshold from OOF cross-validation
  oof_auc            — float
  oof_avg_precision  — float
  cv                 — str, CV type used ("stratified" or "time")
  folds              — int
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
    model: object
    features: list
    medians: dict
    tail_pct: float
    label_on: str
    tail_cut_value: float
    oof_best_threshold: float
    oof_auc: float = float("nan")
    oof_avg_precision: float = float("nan")
    cv: str = "stratified"
    folds: int = 8


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_tail_model(path: str) -> TailModelPack:
    """Load a saved tail model pack from *path* and return a TailModelPack."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Tail model pack not found: {path}")
    raw = joblib.load(path)
    required = {"model", "features", "medians", "tail_pct", "label_on",
                "tail_cut_value", "oof_best_threshold"}
    missing = required - set(raw.keys())
    if missing:
        raise ValueError(f"Tail model pack is missing keys: {missing}")
    return TailModelPack(
        model=raw["model"],
        features=raw["features"],
        medians=raw["medians"],
        tail_pct=raw["tail_pct"],
        label_on=raw["label_on"],
        tail_cut_value=raw["tail_cut_value"],
        oof_best_threshold=raw["oof_best_threshold"],
        oof_auc=raw.get("oof_auc", float("nan")),
        oof_avg_precision=raw.get("oof_avg_precision", float("nan")),
        cv=raw.get("cv", "stratified"),
        folds=raw.get("folds", 8),
    )


def save_tail_model(pack: TailModelPack, path: str) -> None:
    """Save a TailModelPack to *path* as a joblib file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump({
        "model":              pack.model,
        "features":           pack.features,
        "medians":            pack.medians,
        "tail_pct":           pack.tail_pct,
        "label_on":           pack.label_on,
        "tail_cut_value":     pack.tail_cut_value,
        "oof_best_threshold": pack.oof_best_threshold,
        "oof_auc":            pack.oof_auc,
        "oof_avg_precision":  pack.oof_avg_precision,
        "cv":                 pack.cv,
        "folds":              pack.folds,
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

def build_tail_labels(df: pd.DataFrame, label_on: str, tail_pct: float
                      ) -> tuple[np.ndarray, float]:
    """Label the worst tail_pct fraction of trades as 1 (tail loss).

    Args:
        df:       DataFrame with return columns
        label_on: Column to rank by ("return_mon", "return_ann", "return_pct",
                  "return_per_day")
        tail_pct: Fraction to label as tail (e.g. 0.05 for worst 5%)

    Returns:
        (y, tail_cut_value) where y is int array of 0/1 labels
    """
    if label_on not in df.columns:
        raise ValueError(
            f"Column '{label_on}' not found.  Available: {list(df.columns)}"
        )
    target = df[label_on]
    tail_cut = float(target.quantile(tail_pct))
    y = (target <= tail_cut).astype(int).values
    return y, tail_cut


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_tail_data(df: pd.DataFrame, pack: TailModelPack,
                    proba_col: str = "tail_proba") -> tuple[pd.DataFrame, np.ndarray]:
    """Score df with the tail model.  Returns (out_df, proba_array)."""
    X, _ = fill_features(df, pack.features, medians=pack.medians)
    proba = pack.model.predict_proba(X)[:, 1]
    out = df.copy()
    out[proba_col] = proba
    return out, proba


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
