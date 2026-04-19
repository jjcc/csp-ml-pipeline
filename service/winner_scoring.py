#!/usr/bin/env python3
"""
Shared functionality for winner classifier scoring.

This module contains common functions used by both:
- score_winner_classifier_env.py (evaluation/training)
- task_score_tail_winner.py (production scoring)
"""
import json
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Protocol
from dataclasses import dataclass

from sklearn.metrics import average_precision_score, roc_auc_score
from .utils import prep_winner_like_training, pick_threshold_auto

# Module-level constants
DEFAULT_PROBA_COL = "winner_proba"
DEFAULT_PRED_COL = "winner_pred"
DEFAULT_TARGET_COL = "return_mon"
DEFAULT_THRESHOLD = 0.5
DEFAULT_TARGET_EPSILON = 0.0

# Production cleanup columns configuration
PRODUCTION_DROP_COLUMNS = [
    'baseSymbolType', 'expirationDate', 'strike', 'moneyness',
    'breakEvenBid', 'percentToBreakEvenBid', 'tradeTime', 'symbol_norm',
    'impliedVolatilityRank1y', 'delta', 'breakEvenProbability',
    'expirationType', 'symbolType', 'entry_credit', 'exit_intrinsic',
    'total_pnl', 'return_pct', 'ret_2d', 'ret_5d', 'ret_2d_norm',
    'ret_5d_norm', 'prev_close', 'prev_close_minus_ul',
    'prev_close_minus_ul_pct', 'log1p_DTE', 'bid'
]

logger = logging.getLogger(__name__)


class ClassifierModel(Protocol):
    """Protocol for sklearn-like classifier models."""
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        ...


@dataclass
class WinnerModelPack:
    """Container for loaded winner model and metadata."""
    model: ClassifierModel
    features: List[str]
    medians: Optional[Dict[str, float]]
    impute_missing: bool
    best_f1_threshold: float
    label_mode: str = "bins4"          # "bins4" (multiclass) or "binary"
    metrics: Optional[Dict[str, Any]] = None


def load_winner_model(model_path: str) -> WinnerModelPack:
    """
    Load winner classifier model from pickle file.

    Args:
        model_path: Path to the pickled model file

    Returns:
        WinnerModelPack containing model and metadata

    Raises:
        FileNotFoundError: If model file doesn't exist
        KeyError: If required keys are missing from pack
        ValueError: If model_path is empty
    """
    if not model_path:
        raise ValueError("model_path cannot be empty")

    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        pack = joblib.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise

    if "model" not in pack or "features" not in pack:
        raise KeyError("Model pack must contain 'model' and 'features' keys")

    # Simplified impute_missing logic: prefer explicit value, fall back to medians existence
    impute_missing    = pack.get("impute_missing", pack.get("medians") is not None)
    label_mode        = pack.get("label_mode", "bins4")
    metrics = pack.get("metrics", {})
    best_f1_threshold = metrics.get(
        "best_f1_threshold",
        metrics.get("best_f1_threshold_oof", DEFAULT_THRESHOLD),
    )

    return WinnerModelPack(
        model=pack["model"],
        features=pack["features"],
        medians=pack.get("medians"),
        impute_missing=bool(impute_missing),
        best_f1_threshold=float(best_f1_threshold),
        label_mode=str(label_mode),
        metrics=metrics
    )


def score_winner_data(df: pd.DataFrame, model_pack: WinnerModelPack,
                     proba_col: str = DEFAULT_PROBA_COL) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Score data using winner classifier model.

    Args:
        df: DataFrame to score
        model_pack: Loaded model and metadata
        proba_col: Column name for probability scores

    Returns:
        - DataFrame with valid rows (after masking)
        - Probability scores
        - Boolean mask indicating which rows were valid

    Raises:
        ValueError: If DataFrame is empty or proba_col is invalid
    """
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    if not proba_col or not isinstance(proba_col, str):
        raise ValueError("proba_col must be a non-empty string")

    X, mask = prep_winner_like_training(
        df,
        model_pack.features,
        medians=model_pack.medians,
        impute_missing=model_pack.impute_missing
    )

    try:
        all_proba = model_pack.model.predict_proba(X)
    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        raise

    scored_df = df.loc[mask].copy()

    if model_pack.label_mode == "bins4":
        # 4-class multiclass: columns are [p_bin0, p_bin1, p_bin2, p_bin3]
        # Primary signal: p_bin3 (probability of best-quartile outcome)
        for i in range(all_proba.shape[1]):
            scored_df[f"p_bin{i}"] = all_proba[:, i]
        scored_df["conflict_score"] = all_proba[:, 0] * all_proba[:, 3]
        proba = all_proba[:, 3]                    # p_bin3 = win_proba
        scored_df[proba_col] = proba
    else:
        # Binary: index 1 = positive class
        proba = all_proba[:, 1]
        scored_df[proba_col] = proba

    return scored_df, proba, mask


def apply_winner_threshold(df: pd.DataFrame, proba_col: str, pred_col: str,
                          threshold: float) -> pd.DataFrame:
    """
    Apply threshold to probability scores to generate predictions.

    Args:
        df: DataFrame with probability scores
        proba_col: Column containing probabilities
        pred_col: Column name for binary predictions
        threshold: Decision threshold (0-1)

    Returns:
        DataFrame with added prediction column

    Raises:
        ValueError: If threshold is invalid or columns are missing
    """
    if not 0 <= threshold <= 1:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
    if proba_col not in df.columns:
        raise ValueError(f"Column '{proba_col}' not found in DataFrame")

    result = df.copy()
    result[pred_col] = (result[proba_col] >= threshold).astype(int)
    return result


def select_winner_threshold(proba: np.ndarray, y: Optional[np.ndarray],
                           fixed_threshold: Optional[float] = None,
                           use_pack_f1: bool = True,
                           best_f1_threshold: float = DEFAULT_THRESHOLD,
                           auto_calibrate: bool = False,
                           target_precisions: Optional[List[float]] = None,
                           target_recalls: Optional[List[float]] = None) -> Tuple[float, Optional[pd.DataFrame]]:
    """
    Select appropriate threshold based on configuration.

    Priority order:
    1. Fixed threshold (if provided)
    2. Auto-calibrated threshold (if enabled and labels available)
    3. Best F1 threshold from model pack (if use_pack_f1=True)
    4. Default threshold (0.5)

    Args:
        proba: Probability scores
        y: True labels (required for auto-calibration)
        fixed_threshold: Override threshold if provided
        use_pack_f1: Whether to use model's best F1 threshold
        best_f1_threshold: The F1-optimal threshold from training
        auto_calibrate: Enable automatic threshold calibration
        target_precisions: Target precision values for calibration
        target_recalls: Target recall values for calibration

    Returns:
        - Selected threshold
        - Optional threshold table (for auto-calibration)
    """
    # Priority 1: Fixed threshold
    if fixed_threshold is not None:
        return fixed_threshold, None

    # Priority 2: Auto-calibration
    if auto_calibrate and y is not None and (target_precisions or target_recalls):
        return pick_threshold_auto(
            y, proba,
            target_precisions or [],
            target_recalls or []
        )

    # Priority 3: Model's best F1 threshold
    if use_pack_f1:
        return best_f1_threshold, None

    # Priority 4: Default threshold
    return DEFAULT_THRESHOLD, None


def calculate_winner_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    """Calculate standard classification metrics."""
    if len(np.unique(y_true)) <= 1:
        return {"auc_roc": float('nan'), "auc_prc": float('nan')}

    auc_roc = roc_auc_score(y_true, proba)
    auc_prc = average_precision_score(y_true, proba)

    return {"auc_roc": auc_roc, "auc_prc": auc_prc}


def extract_labels(df: pd.DataFrame, train_target: str = DEFAULT_TARGET_COL,
                  train_epsilon: float = DEFAULT_TARGET_EPSILON) -> Optional[np.ndarray]:
    """
    Extract binary labels from DataFrame based on target column.

    Args:
        df: DataFrame containing labels
        train_target: Primary target column name
        train_epsilon: Threshold for converting continuous targets to binary

    Returns:
        Binary label array or None if no valid label column found
    """
    if train_target in df.columns:
        return (pd.to_numeric(df[train_target], errors="coerce") > train_epsilon).values.astype(int)
    elif "win" in df.columns:
        return df["win"].values.astype(int)
    return None


def write_winner_outputs(output_path: str, df: pd.DataFrame, threshold: float,
                        pred_col: str, metrics: Optional[Dict[str, float]] = None,
                        write_metrics_txt: bool = False) -> None:
    """
    Write scoring outputs to files (JSON summary and optional metrics text).

    Args:
        output_path: Base path for output files
        df: Scored DataFrame
        threshold: Applied threshold
        pred_col: Prediction column name
        metrics: Optional metrics dictionary
        write_metrics_txt: Whether to write separate metrics text file

    Raises:
        ValueError: If pred_col is not in DataFrame
        IOError: If file writing fails
    """
    if pred_col not in df.columns:
        raise ValueError(f"Column '{pred_col}' not found in DataFrame")

    output_base = Path(output_path)

    # Prepare summary data
    summary = {
        "rows_scored": int(len(df)),
        "threshold": float(threshold),
        "predicted_winners": int(df[pred_col].sum()),
        "coverage": float(df[pred_col].mean()),
    }

    if metrics:
        summary.update({f"metric_{k}": float(v) for k, v in metrics.items()})

    # Write JSON summary
    json_path = output_base.with_suffix(".json")
    try:
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Wrote summary to {json_path}")
    except Exception as e:
        logger.error(f"Failed to write summary to {json_path}: {e}")
        raise IOError(f"Failed to write summary: {e}") from e

    # Optionally write metrics text file
    if write_metrics_txt and metrics:
        metrics_path = output_base.with_suffix(".metrics.txt")
        try:
            with open(metrics_path, "w") as f:
                for key, value in metrics.items():
                    f.write(f"{key.upper().replace('_', '-')}: {value:.6f}\n")
            logger.info(f"Wrote metrics to {metrics_path}")
        except Exception as e:
            logger.error(f"Failed to write metrics to {metrics_path}: {e}")
            raise IOError(f"Failed to write metrics: {e}") from e


# Backward compatibility aliases
def write_winner_summary(output_path: str, df: pd.DataFrame, threshold: float,
                        pred_col: str, metrics: Optional[Dict[str, float]] = None) -> None:
    """Deprecated: Use write_winner_outputs instead."""
    write_winner_outputs(output_path, df, threshold, pred_col, metrics, write_metrics_txt=False)


def write_winner_metrics(output_path: str, metrics: Dict[str, float]) -> None:
    """Deprecated: Use write_winner_outputs instead."""
    # Create a dummy DataFrame for compatibility
    dummy_df = pd.DataFrame({"dummy_pred": [0]})
    write_winner_outputs(output_path, dummy_df, 0.0, "dummy_pred", metrics, write_metrics_txt=True)


def cleanup_columns_for_production(df: pd.DataFrame, columns_to_drop: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove columns that are not needed in production output.

    Args:
        df: DataFrame to clean
        columns_to_drop: Additional columns to drop beyond defaults

    Returns:
        DataFrame with specified columns removed
    """
    # Start with default production drop list
    drops = PRODUCTION_DROP_COLUMNS.copy()

    # Add GEX columns dynamically
    gex_columns = [col for col in df.columns if col.startswith("gex_")]
    drops.extend(gex_columns)

    # Add any custom drops
    if columns_to_drop:
        drops.extend(columns_to_drop)

    return df.drop(columns=drops, errors='ignore')
