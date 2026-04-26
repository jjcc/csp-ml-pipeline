#!/usr/bin/env python3
"""
b03_train_tail.py — Train the Tail-Loss Classifier (Veto Layer).

The tail classifier is a VETO LAYER that runs AFTER the 4-bin winner classifier.
It identifies "toxic" trades — those the winner model predicted as top-bin (bin-3)
but that actually fall in the worst bin (bin-0).

Architecture dependency:
  b01_train_winner.py must run FIRST and produce winner_scores_oof.csv with the
  4-bin OOF output (columns: row_idx, y_true, y_pred, p_bin0-3, fold, has_oof).
  The tail classifier reuses the same fold structure and uses the per-class
  probabilities (p_bin0-3) as features alongside the standard market features.

Labeling:
  tail = 1  when  return_mon <= quantile(return_mon, tail_k)
  tail_k defaults to 0.05 (worst 5% by absolute monthly return).
  This targets catastrophic absolute losers regardless of daily rank.

High tail_proba => "likely a catastrophic loser — do not enter this trade."

Usage:
    python pipeline/b03_train_tail.py

Outputs (in tail.output_dir from config.yaml):
  tail_classifier_model_{date}.pkl   — TailModelPack (model + calibrator + features + medians)
  tail_scores_oof.csv                — out-of-fold tail probability scores
  tail_classifier_metrics.json       — OOF AUC-ROC, AUC-PRC, threshold, contamination stats
  tail_feature_importances.csv       — feature importances from final model

Configuration comes from config.yaml (tail.*) or .env (TAIL_* prefixed variables).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service.constants import BIN_PROB_FEATS, NEW_GEX_IND_FEATS, TAIL_FEATS, TAIL_EARNINGS_FEATS
from service.env_config import getenv, config as _cfg_loader
from service.tail_scoring import (
    TailModelPack,
    fill_features,
    save_tail_model,
    write_tail_metrics,
)
from service.utils import ensure_dir


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TailClassifierConfig:
    """Parse all training parameters from config.yaml / .env."""

    def __init__(self):
        # Path to winner_scores_oof.csv (bins4 format with p_bin0-3 columns)
        self.winner_oof_csv = getenv("TAIL_WINNER_OOF_CSV", "").strip()
        # Labeled trades CSV (same rolling-window input as b01)
        self.input_csv  = getenv("TAIL_INPUT", "").strip()
        self.output_dir = getenv("TAIL_OUTPUT_DIR", "output/tails_train/").strip()
        self.model_name = getenv("TAIL_MODEL_NAME", "tail_classifier_model").strip()

        self.with_earnings = str(getenv("TAIL_WITH_EARNINGS", "1")).lower() in {
            "1", "true", "yes", "y", "on"
        }
        # Worst-k% absolute quantile cut on return_mon.
        # tail = 1 when return_mon <= quantile(return_mon, tail_k).
        self.tail_k = float(getenv("TAIL_TAIL_K", "0.05"))

        # Optional: GEX indicator folder to merge NEW_GEX_IND_FEATS alongside TAIL_FEATS
        self.gex_folder = getenv("TAIL_GEX_FOLDER", "").strip()

        if not self.winner_oof_csv:
            raise SystemExit(
                "TAIL_WINNER_OOF_CSV must be set in config.yaml (tail.winner_oof_csv).\n"
                "Run b01_train_winner.py first to generate winner_scores_oof.csv."
            )
        if not self.input_csv:
            raise SystemExit("TAIL_INPUT must be set in config.yaml (tail.input).")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_BINS4_OOF_REQUIRED = {"row_idx", "y_true", "y_pred",
                        "p_bin0", "p_bin1", "p_bin2", "p_bin3",
                        "fold", "has_oof"}


def load_and_merge(cfg: TailClassifierConfig) -> pd.DataFrame:
    """Load winner OOF scores + labeled trades and merge on row_idx.

    Returns a DataFrame with all trade features plus the winner OOF columns,
    filtered to rows that have valid OOF predictions (has_oof == 1).
    """
    print(f"[INFO] Loading winner OOF scores: {cfg.winner_oof_csv}")
    if not os.path.isfile(cfg.winner_oof_csv):
        raise FileNotFoundError(
            f"Winner OOF file not found: {cfg.winner_oof_csv}\n"
            "Run b01_train_winner.py first."
        )
    scores = pd.read_csv(cfg.winner_oof_csv)

    missing_cols = _BINS4_OOF_REQUIRED - set(scores.columns)
    if missing_cols:
        raise ValueError(
            f"winner_scores_oof.csv is missing columns: {missing_cols}\n"
            "Ensure b01 was trained with WINNER_LABEL_MODE=bins4."
        )

    print(f"[INFO] Loading labeled trades: {cfg.input_csv}")
    trades = pd.read_csv(cfg.input_csv)

    oof_cols = ["row_idx", "y_true", "y_pred",
                "p_bin0", "p_bin1", "p_bin2", "p_bin3",
                "fold", "has_oof"]
    df = trades.merge(
        scores[oof_cols],
        left_index=True,
        right_on="row_idx",
        how="inner",
    )

    # Keep only rows with valid OOF predictions
    df = df[df["has_oof"] == 1].copy()
    print(f"[INFO] Merged trades with valid OOF: {len(df):,}")

    # Optional: merge GEX indicator features
    if cfg.gex_folder:
        try:
            from pipeline.b13_train_tail_gex import load_gex_indicators, merge_gex_to_trades
            print(f"[INFO] Loading GEX indicators from: {cfg.gex_folder}")
            gex = load_gex_indicators(cfg.gex_folder)
            df = merge_gex_to_trades(df, gex)
            match_rate = df["distance_to_flip"].notna().mean()
            print(f"[INFO] GEX merge: {match_rate:.1%} of training rows matched GEX data")
        except Exception as e:
            print(f"[WARN] GEX merge failed ({e}) — training without GEX features")

    return df


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame,
                   with_earnings: bool) -> tuple[list, pd.DataFrame]:
    """Assemble feature list and add derived features.

    The tail classifier uses:
      - TAIL_FEATS   — market / greeks features
      - BIN_PROB_FEATS — p_bin0-3 from winner OOF + conflict_score
      - TAIL_EARNINGS_FEATS (optional)

    Returns (feat_list, df_with_conflict_score).
    """
    df = df.copy()
    # conflict_score = high when winner model is simultaneously uncertain
    # between best (bin-3) and worst (bin-0) — key tail signal
    df["conflict_score"] = df["p_bin0"] * df["p_bin3"]

    feat_list = list(TAIL_FEATS) + list(BIN_PROB_FEATS)

    if with_earnings:
        earnings_present = [f for f in TAIL_EARNINGS_FEATS if f in df.columns]
        feat_list += earnings_present
        if earnings_present:
            print(f"[INFO] Including earnings features: {earnings_present}")

    # Add GEX indicator features when present (merged by load_and_merge)
    gex_present = [f for f in NEW_GEX_IND_FEATS if f in df.columns]
    if gex_present:
        feat_list += gex_present
        print(f"[INFO] Including {len(gex_present)} GEX indicator features: {gex_present}")

    # Warn about any features the model expects but the data doesn't have
    missing = [f for f in feat_list if f not in df.columns]
    if missing:
        print(f"[WARN] Features missing from data (will impute with median): {missing}")

    return feat_list, df


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
# OOF cross-validation (reuse winner fold structure)
# ---------------------------------------------------------------------------

def run_oof(df: pd.DataFrame, X: pd.DataFrame,
            y: np.ndarray) -> tuple[np.ndarray, list]:
    """Run OOF CV by reusing the winner model's fold column.

    Returns (oof_proba_array, fold_metrics_list).
    """
    oof = np.zeros(len(df), dtype=float)
    fold_metrics = []

    for fold in sorted(df["fold"].unique()):
        if fold == -1:
            continue

        train_mask = (df["fold"] != fold).values
        val_mask   = (df["fold"] == fold).values

        X_train, y_train = X[train_mask], y[train_mask]
        X_val,   y_val   = X[val_mask],   y[val_mask]

        if len(np.unique(y_train)) < 2:
            print(f"  [WARN] Fold {fold}: degenerate label split — skipped")
            continue

        pos = y_train.sum()
        neg = len(y_train) - pos
        scale_pos_weight = neg / max(pos, 1)

        print(f"  Fold {fold}: {len(y_train):,} train "
              f"({y_train.mean():.1%} tail) | "
              f"{len(y_val):,} val ({y_val.mean():.1%} tail) | "
              f"scale_pos_weight={scale_pos_weight:.2f}")

        clf = LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.02,
            num_leaves=64,
            max_depth=-1,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            reg_alpha=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=42 + int(fold),
            verbose=-1,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        oof[val_mask] = clf.predict_proba(X_val)[:, 1]

        if len(np.unique(y_val)) > 1:
            fold_auc = roc_auc_score(y_val, oof[val_mask])
            fold_ap  = average_precision_score(y_val, oof[val_mask])
            print(f"         ROC-AUC={fold_auc:.4f}  PR-AUC={fold_ap:.4f}")
            fold_metrics.append({"fold": int(fold),
                                  "auc": float(fold_auc),
                                  "pr_auc": float(fold_ap)})

    return oof, fold_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = TailClassifierConfig()

    # ── Load and merge winner OOF with labeled trades ────────────────────────
    df = load_and_merge(cfg)

    # ── Feature prep (adds conflict_score, extends with earnings if needed) ──
    feat_list, df = build_features(df, cfg.with_earnings)

    # ── Tail labels: worst-k% by return_mon (absolute quantile cut) ─────────
    # tail = 1 when return_mon <= quantile(return_mon, tail_k).
    # The cut is computed on the training window so no future data leaks in.
    if "return_mon" not in df.columns:
        raise SystemExit(
            "[ERROR] 'return_mon' column not found in training data. "
            "Ensure a04_label_data.py has run and the merged CSV contains return_mon."
        )
    tail_cut  = float(df["return_mon"].quantile(cfg.tail_k))
    y         = (df["return_mon"] <= tail_cut).astype(int).values
    tail_n    = int(y.sum())
    total_n   = len(y)
    tail_rate = float(y.mean())
    print(f"\n[INFO] Tail labels (worst {cfg.tail_k:.0%} by return_mon): "
          f"cut={tail_cut:.4f}  {tail_n}/{total_n} ({tail_rate:.1%} tail)")

    # ── Toxic trade analysis (pred=bin-3 but true=bin-0) ─────────────────────
    mask_pred3 = df["y_pred"] == 3
    mask_toxic = (df["y_pred"] == 3) & (df["y_true"] == 0)
    n_pred3    = int(mask_pred3.sum())
    n_toxic    = int(mask_toxic.sum())
    contamination = n_toxic / max(n_pred3, 1)
    print(f"[INFO] Toxic trades: {n_toxic}/{n_pred3} predicted-bin3 "
          f"are actually bin-0 ({contamination:.1%} contamination rate)")

    if tail_n < 10:
        raise SystemExit(
            f"Too few tail examples ({tail_n}).  "
            "Increase rolling_window_weeks or check winner_scores_oof.csv."
        )

    # ── Build feature matrix ─────────────────────────────────────────────────
    print(f"\n[INFO] Building feature matrix: {len(feat_list)} features")
    X, medians = fill_features(df, feat_list)

    # ── OOF cross-validation (reuse winner fold structure) ───────────────────
    print(f"\n[INFO] OOF CV (reusing winner fold structure)…")
    oof_proba, fold_metrics = run_oof(df, X, y)

    valid_mask = oof_proba > 0
    if valid_mask.sum() > 0 and len(np.unique(y[valid_mask])) > 1:
        oof_auc = roc_auc_score(y[valid_mask], oof_proba[valid_mask])
        oof_ap  = average_precision_score(y[valid_mask], oof_proba[valid_mask])
        best_thr = find_best_f1_threshold(y[valid_mask], oof_proba[valid_mask])
    else:
        oof_auc  = float("nan")
        oof_ap   = float("nan")
        best_thr = 0.5

    print(f"\n[INFO] OOF: AUC-ROC={oof_auc:.4f}  AUC-PRC={oof_ap:.4f}  "
          f"best_threshold={best_thr:.4f}")

    # ── Isotonic calibration on OOF probabilities ────────────────────────────
    print(f"[INFO] Calibrating OOF probabilities (IsotonicRegression)…")
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(oof_proba[valid_mask], y[valid_mask])

    # ── Final model: train on all data ───────────────────────────────────────
    print(f"\n[INFO] Training final model on all {total_n:,} trades…")
    pos = y.sum()
    neg = total_n - pos
    final_model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=3.0,
        reg_alpha=1.0,
        scale_pos_weight=neg / max(pos, 1),
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    final_model.fit(X, y)

    # ── Output directory ─────────────────────────────────────────────────────
    score_date = _cfg_loader.get_score_date()
    out_dir = cfg.output_dir.rstrip("/\\")
    ensure_dir(out_dir)

    # ── Save TailModelPack ───────────────────────────────────────────────────
    pack = TailModelPack(
        model=final_model,
        calibrator=cal,
        features=feat_list,
        medians=medians,
        oof_best_threshold=best_thr,
        oof_auc=oof_auc,
        oof_avg_precision=oof_ap,
        tail_rate=tail_rate,
        contamination_rate=contamination,
    )
    model_fname = f"{cfg.model_name}_{score_date}.pkl"
    model_path  = os.path.join(out_dir, model_fname)
    save_tail_model(pack, model_path)
    print(f"[INFO] Model pack → {model_path}")

    # ── Feature importances ──────────────────────────────────────────────────
    imp_path = os.path.join(out_dir, "tail_feature_importances.csv")
    pd.DataFrame({
        "feature":    feat_list,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False).to_csv(imp_path, index=False)
    print(f"[INFO] Feature importances → {imp_path}")

    # ── OOF scores CSV ───────────────────────────────────────────────────────
    id_cols = ["baseSymbol", "tradeTime", "expirationDate", "strike",
               "potentialReturnAnnual", "return_mon", "daysToExpiration",
               "VIX", "impliedVolatilityRank1y"]
    have = [c for c in id_cols if c in df.columns]
    oof_df = df[have].copy()
    oof_df["row_idx"]        = df["row_idx"].values
    oof_df["fold"]           = df["fold"].values
    oof_df["y_true_bin"]     = df["y_true"].values    # 0-3 from winner bins4
    oof_df["y_pred_bin"]     = df["y_pred"].values
    oof_df["is_tail"]        = y
    oof_df["tail_proba_oof"] = oof_proba
    oof_df["tail_proba_cal"] = cal.transform(oof_proba)
    oof_df["is_toxic"]       = mask_toxic.values

    oof_path = os.path.join(out_dir, "tail_scores_oof.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"[INFO] OOF scores → {oof_path}")

    # ── Metrics JSON ─────────────────────────────────────────────────────────
    write_tail_metrics(out_dir, {"auc_roc": oof_auc, "auc_prc": oof_ap}, extra={
        "rows":               total_n,
        "tail_k":             cfg.tail_k,
        "tail_cut_return_mon": round(tail_cut, 6),
        "tail_n":             tail_n,
        "tail_rate":          round(tail_rate, 4),
        "oof_best_threshold": round(best_thr, 6),
        "contamination_rate": round(contamination, 4),
        "toxic_n":            n_toxic,
        "pred_bin3_n":        n_pred3,
        "fold_metrics":       fold_metrics,
        "model_path":         model_path,
        "winner_oof_csv":     cfg.winner_oof_csv,
    })

    print(f"\n✅  Tail classifier trained.")
    print(f"   ROC AUC (OOF)={oof_auc:.4f}   PR AUC (OOF)={oof_ap:.4f}")
    print(f"   Tail rate={tail_rate:.1%}  Contamination={contamination:.1%}")
    print(f"   Outputs → {out_dir}/")
    print(f"\n   Next: python pipeline/b04_score_tail.py")


if __name__ == "__main__":
    main()
