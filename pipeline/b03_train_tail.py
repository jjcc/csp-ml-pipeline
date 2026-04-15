#!/usr/bin/env python3
"""
b03_train_tail.py — Train the Tail-Loss Classifier with OOF cross-validation.

Trains a binary classifier to identify the worst-K% of CSP trades by monthly
return — contracts that are likely to produce catastrophic losses ("fat tails").
High tail_proba ≈ likely loser; filter these out before entering a trade.

Training input:  same rolling-window merged CSV as b01 (output of a05)
Labeling:        worst `tail.pct` fraction of trades by `tail.label_on` return
Model:           GradientBoostingClassifier (sklearn) — robust on small datasets
                 or LightGBM (lgbm) for larger windows
CV:              StratifiedKFold (default) or TimeSeriesSplit

Outputs (in tail.output_dir from config.yaml):
  tail_classifier_model_{date}.pkl   — model pack (model + features + medians + metrics)
  tail_scores_oof.csv                — out-of-fold probability scores (sampled)
  tail_classifier_metrics.json       — AUC-ROC, AUC-PR, best-F1 threshold, etc.
  tail_feature_importances.csv       — mean feature importances across folds

Configuration comes from config.yaml (tail.*) or .env (TAIL_* prefixed variables).

Usage:
    python pipeline/b03_train_tail.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service.constants import TAIL_FEATS, TAIL_EARNINGS_FEATS
from service.env_config import getenv, config as _cfg_loader
from service.preprocess import add_dte_and_normalized_returns
from service.tail_scoring import (
    TailModelPack, build_tail_labels, fill_features, save_tail_model,
    write_tail_metrics,
)
from service.utils import ensure_dir, prep_tail_training_df


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TailClassifierConfig:
    """Parse all training parameters from config.yaml / .env."""

    def __init__(self):
        self.input_csv  = getenv("TAIL_INPUT", "")
        self.output_dir = getenv("TAIL_OUTPUT_DIR", "output/tails_train/")
        self.model_name = getenv("TAIL_MODEL_NAME", "tail_classifier_model")

        self.tail_pct   = float(getenv("TAIL_PCT",    "0.05"))
        self.label_on   = getenv("TAIL_LABEL_ON",     "return_mon").strip()
        self.cv_type    = getenv("TAIL_CV_TYPE",       "stratified").strip().lower()
        self.folds      = int(getenv("TAIL_FOLDS",    "8"))
        self.seed       = int(getenv("TAIL_SEED",     "42"))
        self.model_type = getenv("TAIL_MODEL_TYPE",   "gbm").strip().lower()
        self.with_earnings = str(getenv("TAIL_WITH_EARNINGS", "1")).lower() in {
            "1", "true", "yes", "y", "on"
        }
        self.save_oof_sample = int(getenv("TAIL_SAVE_SCORES_SAMPLE", "40000"))

        if not self.input_csv:
            raise SystemExit("TAIL_INPUT must be set in config.yaml.")
        if not self.output_dir:
            raise SystemExit("TAIL_OUTPUT_DIR must be set in config.yaml.")


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: TailClassifierConfig):
    """Return an untrained classifier based on cfg.model_type."""
    if cfg.model_type == "lgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            raise SystemExit("LightGBM not installed.  Run: pip install lightgbm")
        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=cfg.seed,
            n_jobs=-1,
            verbose=-1,
        )
    else:  # default: gradient boosting
        return GradientBoostingClassifier(random_state=cfg.seed)


# ---------------------------------------------------------------------------
# OOF cross-validation
# ---------------------------------------------------------------------------

def run_oof(X: np.ndarray, y: np.ndarray, cfg: TailClassifierConfig,
            feat_list: list) -> tuple[np.ndarray, np.ndarray]:
    """Run OOF cross-validation.  Returns (oof_proba, mean_importances)."""
    if cfg.cv_type == "time":
        splitter = TimeSeriesSplit(n_splits=cfg.folds)
        splits = list(splitter.split(X))
    else:
        splitter = StratifiedKFold(n_splits=cfg.folds, shuffle=True,
                                   random_state=cfg.seed)
        splits = list(splitter.split(X, y))

    oof = np.zeros(len(y), dtype=float)
    importances = np.zeros(len(feat_list), dtype=float)
    n_valid = 0

    for tr_idx, va_idx in splits:
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]

        if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
            print(f"  [WARN] Fold skipped — degenerate label split")
            continue

        clf = build_model(cfg)
        clf.fit(Xtr, ytr)
        oof[va_idx] = clf.predict_proba(Xva)[:, 1]
        if hasattr(clf, "feature_importances_"):
            importances += clf.feature_importances_
        n_valid += 1

    if n_valid == 0:
        raise SystemExit(
            "All OOF folds were degenerate.  "
            f"Increase tail.pct (currently {cfg.tail_pct:.3f}) or use more data."
        )

    importances /= n_valid
    return oof, importances


# ---------------------------------------------------------------------------
# Best-F1 threshold
# ---------------------------------------------------------------------------

def find_best_f1_threshold(y: np.ndarray, proba: np.ndarray) -> float:
    """Return the threshold that maximises F1 on the OOF predictions."""
    prec, rec, thr = precision_recall_curve(y, proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = int(np.nanargmax(f1))
    if 0 < best_idx < len(thr):
        return float(thr[best_idx - 1])
    return 0.5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = TailClassifierConfig()

    # --- Load and prep data ---
    print(f"[INFO] Loading training data from {cfg.input_csv}")
    df = pd.read_csv(cfg.input_csv)
    df = prep_tail_training_df(df).dropna(subset=["total_pnl"])
    df = add_dte_and_normalized_returns(df)
    df = df.sort_values("tradeTime").reset_index(drop=True)
    print(f"[INFO] {len(df):,} rows after prep")

    # --- Feature list ---
    feat_list = list(TAIL_FEATS)
    if cfg.with_earnings:
        earnings_present = [f for f in TAIL_EARNINGS_FEATS if f in df.columns]
        feat_list += earnings_present
        if earnings_present:
            print(f"[INFO] Including earnings features: {earnings_present}")

    # --- Build X and labels ---
    X, medians = fill_features(df, feat_list)
    y, tail_cut = build_tail_labels(df, cfg.label_on, cfg.tail_pct)

    print(f"[INFO] Labeling: {cfg.label_on} ≤ {tail_cut:.4f} "
          f"→ {y.sum()} tails out of {len(y)} ({y.mean()*100:.1f}%)")

    if y.sum() < max(5, cfg.folds):
        raise SystemExit(
            f"Not enough tail examples ({y.sum()}) for {cfg.folds} folds.  "
            f"Increase tail.pct or use more training data."
        )

    # --- OOF cross-validation ---
    print(f"[INFO] OOF CV: {cfg.cv_type}, {cfg.folds} folds, model={cfg.model_type}")
    t0 = pd.Timestamp.now()
    oof_proba, importances = run_oof(X, y, cfg, feat_list)
    elapsed = (pd.Timestamp.now() - t0).total_seconds()
    print(f"[INFO] OOF done in {elapsed:.1f}s")

    # --- OOF metrics ---
    oof_auc = roc_auc_score(y, oof_proba) if len(np.unique(y)) > 1 else float("nan")
    oof_ap  = average_precision_score(y, oof_proba) if len(np.unique(y)) > 1 else float("nan")
    best_thr = find_best_f1_threshold(y, oof_proba)
    print(f"[INFO] OOF  AUC-ROC={oof_auc:.4f}  AUC-PRC={oof_ap:.4f}  "
          f"best_threshold≈{best_thr:.4f}")

    # --- Final model fit on all data ---
    clf_final = build_model(cfg)
    clf_final.fit(X, y)

    # --- Output directory and model identifier ---
    score_date = _cfg_loader.get_score_date()   # e.g. "20251027"
    out_dir = cfg.output_dir.rstrip("/\\")
    ensure_dir(out_dir)
    model_fname = f"{cfg.model_name}_{score_date}.pkl"
    model_path  = os.path.join(out_dir, model_fname)

    # --- Save model pack ---
    pack = TailModelPack(
        model=clf_final,
        features=feat_list,
        medians=medians,
        tail_pct=cfg.tail_pct,
        label_on=cfg.label_on,
        tail_cut_value=tail_cut,
        oof_best_threshold=best_thr,
        oof_auc=oof_auc,
        oof_avg_precision=oof_ap,
        cv=cfg.cv_type,
        folds=cfg.folds,
    )
    save_tail_model(pack, model_path)
    print(f"[INFO] Model pack → {model_path}")

    # --- Feature importances ---
    imp_path = os.path.join(out_dir, "tail_feature_importances.csv")
    pd.DataFrame({"feature": feat_list, "importance": importances}) \
        .sort_values("importance", ascending=False) \
        .to_csv(imp_path, index=False)
    print(f"[INFO] Feature importances → {imp_path}")

    # --- OOF scores (sampled) ---
    if cfg.save_oof_sample > 0:
        oof_cols = ["baseSymbol", "tradeTime", "expirationDate", "strike",
                    "potentialReturnAnnual", "total_pnl",
                    "return_pct", "return_mon", "return_ann", "daysToExpiration",
                    "VIX", "impliedVolatilityRank1y",
                    "gex_gamma_at_ul", "gex_total_abs", "gex_neg", "gex_missing",
                    "prev_close_minus_ul_pct", "log1p_DTE"]
        have = [c for c in oof_cols if c in df.columns]
        oof_df = df[have].copy()
        oof_df["tail_proba_oof"] = oof_proba
        oof_df["is_tail"] = y
        if len(oof_df) > cfg.save_oof_sample:
            oof_df = oof_df.sample(cfg.save_oof_sample,
                                   random_state=cfg.seed).sort_values("tradeTime")
        oof_path = os.path.join(out_dir, "tail_scores_oof.csv")
        oof_df.to_csv(oof_path, index=False)
        print(f"[INFO] OOF scores → {oof_path}")

    # --- Metrics JSON ---
    write_tail_metrics(out_dir, {"auc_roc": oof_auc, "auc_prc": oof_ap}, extra={
        "rows":            int(len(df)),
        "tails":           int(y.sum()),
        "tail_pct":        float(cfg.tail_pct),
        "label_on":        cfg.label_on,
        "tail_cut_value":  float(tail_cut),
        "oof_best_threshold": float(best_thr),
        "cv":              cfg.cv_type,
        "folds":           cfg.folds,
        "model_type":      cfg.model_type,
        "model_path":      model_path,
    })

    print(f"\n✅  Tail classifier trained.")
    print(f"   ROC AUC (OOF)={oof_auc:.4f}   PR AUC (OOF)={oof_ap:.4f}")
    print(f"   Tail fraction={cfg.tail_pct:.3f}  label={cfg.label_on}  "
          f"cut={tail_cut:.4f}")
    print(f"   Outputs → {out_dir}/")


if __name__ == "__main__":
    main()
