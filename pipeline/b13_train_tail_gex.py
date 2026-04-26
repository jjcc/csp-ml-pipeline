#!/usr/bin/env python3
"""
b13_train_tail_gex.py — Train GEX-Only Tail Classifier.

Standalone process — no dependency on b01/b03.

Uses only the new GEX indicator features from NEW_GEX_FEATURE_FOLDER:
  gamma_flip, distance_to_flip, flip_score, gamma_density_near,
  gamma_density_below, support_asymmetry, downside_void, peak_concentration,
  slope_near_price, negative_gamma_below_ratio, has_flip, no_flip_flag, regime.

GEX alignment: capture_dt <= tradeTime (backward asof merge per symbol).
Labels: tail = 1 when return_mon <= quantile(return_mon, tail_k).
CV: TimeSeriesSplit(5) on sorted tradeTime.

Usage:
    python pipeline/b13_train_tail_gex.py

Outputs (in tail_gex.output_dir from config.yaml):
  tail_gex_model_{date}.pkl          — model pack (model + calibrator + features + medians)
  tail_gex_scores_oof.csv            — out-of-fold tail probabilities
  tail_gex_metrics.json              — OOF AUC-ROC, AUC-PRC, threshold, tail rate
  tail_gex_feature_importances.csv   — feature importances from final model
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service.constants import NEW_GEX_IND_FEATS
from service.env_config import getenv, config as _cfg_loader
from service.tail_scoring import fill_features, write_tail_metrics
from service.utils import ensure_dir


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class GexTailConfig:
    def __init__(self):
        self.input_csv  = getenv("TAIL_GEX_INPUT", "").strip()
        self.output_dir = getenv("TAIL_GEX_OUTPUT_DIR", "output/tails_gex_train/").strip()
        self.model_name = getenv("TAIL_GEX_MODEL_NAME", "tail_gex_model").strip()
        self.tail_k     = float(getenv("TAIL_GEX_TAIL_K", "0.05"))
        # gex_folder: config.yaml tail_gex.gex_folder, else .env NEW_GEX_FEATURE_FOLDER
        gex_folder = getenv("TAIL_GEX_GEX_FOLDER", "").strip()
        if not gex_folder:
            gex_folder = getenv("NEW_GEX_FEATURE_FOLDER", "").strip()
        self.gex_folder = gex_folder

        if not self.input_csv:
            raise SystemExit("TAIL_GEX_INPUT must be set (tail_gex.input in config.yaml).")
        if not self.gex_folder:
            raise SystemExit(
                "GEX folder not configured.  Set NEW_GEX_FEATURE_FOLDER in .env "
                "or tail_gex.gex_folder in config.yaml."
            )


# ---------------------------------------------------------------------------
# GEX data loading
# ---------------------------------------------------------------------------

def load_gex_indicators(gex_folder: str) -> pd.DataFrame:
    """Load all monthly parquet files from gex_folder; keep one snapshot per (symbol, date) near 11:00."""
    folder = Path(gex_folder)
    paths = sorted(folder.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found in GEX folder: {gex_folder}")

    gex = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)

    gex["capture_dt"] = pd.to_datetime(gex["capture_dt"], errors="coerce").astype("datetime64[us]")
    gex = gex.dropna(subset=["capture_dt", "symbol"]).copy()
    # has_flip is bool in parquet — cast to float so fill_features median works cleanly
    if "has_flip" in gex.columns:
        gex["has_flip"] = gex["has_flip"].astype(float)

    # Keep one snapshot per (symbol, date): closest to 11:00
    TARGET_MIN = 11 * 60
    gex["_date"] = gex["capture_dt"].dt.date
    gex["_dist"] = (gex["capture_dt"].dt.hour * 60 + gex["capture_dt"].dt.minute - TARGET_MIN).abs()
    gex = (gex.sort_values("_dist")
              .groupby(["symbol", "_date"], as_index=False)
              .first()
              .drop(columns=["_date", "_dist"]))

    print(f"[INFO] GEX indicators loaded: {len(gex):,} rows, "
          f"{gex['symbol'].nunique()} symbols, "
          f"{gex['capture_dt'].dt.date.nunique()} dates")
    return gex.sort_values(["symbol", "capture_dt"]).reset_index(drop=True)


def merge_gex_to_trades(trades: pd.DataFrame, gex: pd.DataFrame) -> pd.DataFrame:
    """Backward asof merge per symbol: GEX.capture_dt <= trade.tradeTime.

    merge_asof requires global monotonicity of the key column, so we merge
    per symbol and concat rather than using the `by` parameter.
    """
    trades = trades.copy()
    trades["tradeTime"] = pd.to_datetime(trades["tradeTime"], errors="coerce").astype("datetime64[us]")

    gex_by_sym = {sym: grp.sort_values("capture_dt")
                  for sym, grp in gex.groupby("symbol")}

    pieces = []
    for sym, sym_trades in trades.groupby("baseSymbol"):
        sym_trades_sorted = sym_trades.sort_values("tradeTime")
        sym_gex = gex_by_sym.get(sym)
        if sym_gex is None or sym_gex.empty:
            pieces.append(sym_trades_sorted)
            continue
        merged = pd.merge_asof(
            sym_trades_sorted,
            sym_gex.rename(columns={"symbol": "baseSymbol"}),
            left_on="tradeTime",
            right_on="capture_dt",
            direction="backward",
        )
        pieces.append(merged)

    result = pd.concat(pieces, ignore_index=True) if pieces else trades.iloc[0:0]

    n_matched = result["distance_to_flip"].notna().sum() if "distance_to_flip" in result.columns else 0
    print(f"[INFO] GEX merge: {len(result):,} trades, "
          f"{n_matched:,} matched ({n_matched/max(len(result),1):.1%})")
    return result


# ---------------------------------------------------------------------------
# Feature prep
# ---------------------------------------------------------------------------

def build_feat_list() -> list[str]:
    """Return the GEX feature list.  regime is already float64 (1.0/-1.0) — no encoding needed."""
    return list(NEW_GEX_IND_FEATS)


# ---------------------------------------------------------------------------
# OOF cross-validation
# ---------------------------------------------------------------------------

def run_oof(df: pd.DataFrame, X: pd.DataFrame, y: np.ndarray,
            n_splits: int = 5) -> tuple[np.ndarray, list]:
    """Walk-forward OOF via TimeSeriesSplit (sorts on tradeTime implicitly via df order)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof  = np.zeros(len(df), dtype=float)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val,   y_val   = X.iloc[val_idx],   y[val_idx]

        if len(np.unique(y_train)) < 2:
            print(f"  [WARN] Fold {fold}: degenerate label split — skipped")
            continue

        pos = y_train.sum()
        neg = len(y_train) - pos
        spw = neg / max(pos, 1)

        print(f"  Fold {fold}: {len(y_train):,} train ({y_train.mean():.1%} tail) | "
              f"{len(y_val):,} val ({y_val.mean():.1%} tail) | scale_pos_weight={spw:.2f}")

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
            scale_pos_weight=spw,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        oof[val_idx] = clf.predict_proba(X_val)[:, 1]

        if len(np.unique(y_val)) > 1:
            fold_auc = roc_auc_score(y_val, oof[val_idx])
            fold_ap  = average_precision_score(y_val, oof[val_idx])
            print(f"         ROC-AUC={fold_auc:.4f}  PR-AUC={fold_ap:.4f}")
            fold_metrics.append({"fold": fold, "auc": float(fold_auc), "pr_auc": float(fold_ap)})

    return oof, fold_metrics


# ---------------------------------------------------------------------------
# Best-F1 threshold
# ---------------------------------------------------------------------------

def find_best_f1_threshold(y: np.ndarray, proba: np.ndarray) -> float:
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
    cfg = GexTailConfig()

    # ── Load labeled trades ───────────────────────────────────────────────────
    print(f"[INFO] Loading labeled trades: {cfg.input_csv}")
    trades = pd.read_csv(cfg.input_csv)
    print(f"[INFO] {len(trades):,} trades loaded")

    # ── Load and merge GEX indicators ─────────────────────────────────────────
    gex = load_gex_indicators(cfg.gex_folder)
    df  = merge_gex_to_trades(trades, gex)

    # ── Tail labels ───────────────────────────────────────────────────────────
    if "return_mon" not in df.columns:
        raise SystemExit("[ERROR] 'return_mon' not found — run a04_label_data.py first.")
    tail_cut  = float(df["return_mon"].quantile(cfg.tail_k))
    y         = (df["return_mon"] <= tail_cut).astype(int).values
    tail_n    = int(y.sum())
    total_n   = len(y)
    tail_rate = float(y.mean())
    print(f"\n[INFO] Tail labels (worst {cfg.tail_k:.0%} by return_mon): "
          f"cut={tail_cut:.4f}  {tail_n}/{total_n} ({tail_rate:.1%} tail)")

    if tail_n < 10:
        raise SystemExit(f"Too few tail examples ({tail_n}).  "
                         "Increase rolling_window_weeks or check tail_k.")

    # ── Feature matrix ────────────────────────────────────────────────────────
    feat_list = build_feat_list()
    print(f"\n[INFO] Building feature matrix: {feat_list}")

    # Sort by tradeTime for TimeSeriesSplit ordering
    df = df.sort_values("tradeTime").reset_index(drop=True)
    y  = (df["return_mon"] <= tail_cut).astype(int).values

    X, medians = fill_features(df, feat_list)

    # ── OOF cross-validation ──────────────────────────────────────────────────
    print(f"\n[INFO] OOF CV (TimeSeriesSplit, 5 folds)…")
    oof_proba, fold_metrics = run_oof(df, X, y)

    valid_mask = oof_proba > 0
    if valid_mask.sum() > 0 and len(np.unique(y[valid_mask])) > 1:
        oof_auc  = roc_auc_score(y[valid_mask], oof_proba[valid_mask])
        oof_ap   = average_precision_score(y[valid_mask], oof_proba[valid_mask])
        best_thr = find_best_f1_threshold(y[valid_mask], oof_proba[valid_mask])
    else:
        oof_auc  = float("nan")
        oof_ap   = float("nan")
        best_thr = 0.5

    print(f"\n[INFO] OOF: AUC-ROC={oof_auc:.4f}  AUC-PRC={oof_ap:.4f}  "
          f"best_threshold={best_thr:.4f}")

    # ── Isotonic calibration ──────────────────────────────────────────────────
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(oof_proba[valid_mask], y[valid_mask])

    # ── Final model on all data ───────────────────────────────────────────────
    print(f"\n[INFO] Training final model on all {total_n:,} trades…")
    pos = y.sum(); neg = total_n - pos
    final_model = LGBMClassifier(
        n_estimators=2000, learning_rate=0.02, num_leaves=64, max_depth=-1,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=3.0, reg_alpha=1.0, scale_pos_weight=neg / max(pos, 1),
        random_state=42, verbose=-1, n_jobs=-1,
    )
    final_model.fit(X, y)

    # ── Output directory ──────────────────────────────────────────────────────
    score_date = _cfg_loader.get_score_date()
    out_dir = cfg.output_dir.rstrip("/\\")
    ensure_dir(out_dir)

    # ── Save model pack ───────────────────────────────────────────────────────
    model_fname = f"{cfg.model_name}_{score_date}.pkl"
    model_path  = os.path.join(out_dir, model_fname)
    joblib.dump({
        "model":               final_model,
        "calibrator":          cal,
        "features":            feat_list,
        "medians":             medians,
        "oof_best_threshold":  best_thr,
        "oof_auc":             oof_auc,
        "oof_avg_precision":   oof_ap,
        "tail_rate":           tail_rate,
        "contamination_rate":  0.0,   # N/A for standalone GEX model
        "gex_folder":          cfg.gex_folder,
    }, model_path)
    print(f"[INFO] Model pack → {model_path}")

    # ── Feature importances ───────────────────────────────────────────────────
    imp_path = os.path.join(out_dir, "tail_gex_feature_importances.csv")
    pd.DataFrame({
        "feature":    feat_list,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False).to_csv(imp_path, index=False)
    print(f"[INFO] Feature importances → {imp_path}")

    # ── OOF scores CSV ────────────────────────────────────────────────────────
    id_cols = ["baseSymbol", "tradeTime", "expirationDate", "strike",
               "potentialReturnAnnual", "return_mon", "daysToExpiration"]
    have = [c for c in id_cols if c in df.columns]
    oof_df = df[have].copy()
    oof_df["is_tail"]        = y
    oof_df["tail_gex_proba_oof"] = oof_proba
    oof_df["tail_gex_proba_cal"] = cal.transform(oof_proba)

    oof_path = os.path.join(out_dir, "tail_gex_scores_oof.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"[INFO] OOF scores → {oof_path}")

    # ── Metrics JSON ──────────────────────────────────────────────────────────
    write_tail_metrics(out_dir, {"auc_roc": oof_auc, "auc_prc": oof_ap}, extra={
        "rows":               total_n,
        "tail_k":             cfg.tail_k,
        "tail_cut_return_mon": round(tail_cut, 6),
        "tail_n":             tail_n,
        "tail_rate":          round(tail_rate, 4),
        "oof_best_threshold": round(best_thr, 6),
        "fold_metrics":       fold_metrics,
        "model_path":         model_path,
        "gex_folder":         cfg.gex_folder,
        "features":           feat_list,
    })

    print(f"\n✅  GEX-only tail classifier trained.")
    print(f"   ROC AUC (OOF)={oof_auc:.4f}   PR AUC (OOF)={oof_ap:.4f}")
    print(f"   Tail rate={tail_rate:.1%}")
    print(f"   Outputs → {out_dir}/")
    print(f"\n   Next: python pipeline/b14_score_tail_gex.py")


if __name__ == "__main__":
    main()
