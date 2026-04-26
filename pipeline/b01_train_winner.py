#!/usr/bin/env python3
"""
b01_train_winner.py — Train the Winner Classifier with OOF cross-validation.

Trains a binary classifier to predict profitable Cash-Secured Put trades.
Supports LightGBM (default), CatBoost, and RandomForest backends.

Outputs (in winner.output_dir from config.yaml):
  winner_classifier_model_<tag>_<type>.pkl  — model pack (model + features + medians + metrics)
  winner_scores_oof.csv                     — out-of-fold probability scores
  winner_classifier_metrics.json            — AUC-ROC, AUC-PR, best F1 threshold, etc.
  threshold_table.csv                       — thresholds at requested precision/recall targets
  precision_recall_coverage.csv/.png        — PR-coverage curves

Configuration comes from config.yaml (winner.*) or .env (WINNER_* prefixed variables).

Usage:
    python pipeline/b01_train_winner.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
import joblib

# Ensure project root is on path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service.constants import BASE_FEATS, NEW_FEATS, NEW_GEX_IND_FEATS
from service.preprocess import add_dte_and_normalized_returns
from service.env_config import getenv, config as _cfg_loader
from service.table_store import read_table, table_exists
from service.utils import ensure_dir


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class WinnerClassifierConfig:
    """Parse all training parameters from config.yaml / .env."""

    def __init__(self):
        self._parse()

    # --- parsing helpers ---
    @staticmethod
    def _list_float(val: str) -> List[float]:
        if not val or not val.strip():
            return []
        try:
            arr = json.loads(val.strip())
            if isinstance(arr, list):
                return [float(x) for x in arr]
        except Exception:
            pass
        return [float(x.strip()) for x in val.split(",") if x.strip()]

    @staticmethod
    def _list_str(val: str) -> List[str]:
        if not val or not val.strip():
            return []
        try:
            arr = json.loads(val.strip())
            if isinstance(arr, list):
                return [str(x) for x in arr]
        except Exception:
            pass
        return [x.strip() for x in val.split(",") if x.strip()]

    @staticmethod
    def _maybe_none(val: str):
        return None if not val or not str(val).strip() else val

    @staticmethod
    def _bool(val: str, default: bool = False) -> bool:
        if val is None:
            return default
        return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _parse(self):
        # I/O — WINNER_INPUT is the fully-resolved path from config.yaml winner.input
        self.input_csv  = getenv("WINNER_INPUT", "")
        self.output_dir = getenv("WINNER_OUTPUT_DIR", "output/winner_train/")
        self.model_name = getenv("WINNER_MODEL_NAME", "winner_classifier_model")

        # Feature selection (subset of ALL_FEATS used by default)
        gex_subset      = ["gex_neg", "gex_center_abs_strike", "gex_total_abs"]
        self.features   = BASE_FEATS + NEW_FEATS + gex_subset
        # Optional: GEX indicator folder — NEW_GEX_IND_FEATS appended in main() after merge
        self.gex_folder = getenv("WINNER_GEX_FOLDER", "").strip()
        self.id_cols    = self._list_str(getenv("WINNER_ID_COLS", ""))

        # Model
        self.random_state      = int(getenv("WINNER_RANDOM_STATE",           "42"))
        self.n_estimators      = int(getenv("WINNER_CLASSIFIER_N_ESTIMATORS","400"))
        self.class_weight      = self._maybe_none(getenv("WINNER_CLASS_WEIGHT","balanced_subsample"))
        self.max_depth         = None if not getenv("WINNER_MAX_DEPTH","").strip() else int(getenv("WINNER_MAX_DEPTH"))
        self.min_samples_leaf  = int(getenv("WINNER_MIN_SAMPLES_LEAF",  "1"))
        self.min_samples_split = int(getenv("WINNER_MIN_SAMPLES_SPLIT", "2"))
        self.model_type        = getenv("WINNER_MODEL_TYPE", "lgbm").lower()

        # Preprocessing
        self.impute_missing = self._bool(getenv("WINNER_IMPUTE_MISSING", "1"))
        self.use_weights    = self._bool(getenv("WINNER_USE_WEIGHTS",    "1"))
        self.weight_alpha   = float(getenv("WINNER_WEIGHT_ALPHA", "0.02"))
        self.weight_min     = float(getenv("WINNER_WEIGHT_MIN",   "0.5"))
        self.weight_max     = float(getenv("WINNER_WEIGHT_MAX",   "10.0"))

        # Training target
        self.train_target = getenv("WINNER_TRAIN_TARGET", "return_mon").strip()

        # Cross-validation
        self.oof_folds    = int(getenv("WINNER_OOF_FOLDS",    "5"))
        self.time_series  = getenv("WINNER_TIME_SERIES", "auto").strip().lower()

        # Early stopping (gradient boosting only)
        self.early_stopping_rounds = int(getenv("WINNER_EARLY_STOPPING_ROUNDS", "100"))
        self.valid_fraction        = float(getenv("WINNER_VALID_FRACTION", "0.1"))

        # Evaluation targets
        self.targets_recall    = self._list_float(getenv("WINNER_TARGET_RECALL",    ""))
        self.targets_precision = self._list_float(getenv("WINNER_TARGET_PRECISION", ""))

        # --- 4-bin classification (default mode) ---
        # bins4: labels trades into 4 return quartiles (0=worst, 3=best)
        # binary: original win/loss binary label (return_mon > epsilon)
        self.label_mode      = getenv("WINNER_LABEL_MODE", "bins4").strip().lower()
        self.bins_mode       = getenv("WINNER_BINS_MODE",  "per_day").strip().lower()
        raw_q = getenv("WINNER_BINS_Q", "[0.25,0.5,0.75]").strip()
        try:
            import json as _json
            self.bins_q = [float(x) for x in _json.loads(raw_q)]
        except Exception:
            self.bins_q = [float(x.strip()) for x in raw_q.strip("[]").split(",") if x.strip()]
        self.bins_min_group  = int(getenv("WINNER_BINS_MIN_GROUP", "20"))
        self.time_col        = getenv("WINNER_TIME_COL", "captureTime").strip()

        if not self.input_csv or not self.output_dir:
            raise SystemExit("WINNER_INPUT and WINNER_OUTPUT_DIR must be set in config.yaml.")


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def select_features(df: pd.DataFrame, explicit: List[str], id_cols: List[str]) -> List[str]:
    """Return explicit list (validated) or auto-detect numeric columns."""
    if explicit:
        missing = [c for c in explicit if c not in df.columns]
        if missing:
            raise ValueError(f"Requested features not found in data: {missing}")
        return explicit
    exclude = set(id_cols or []) | {"return_pct", "return_mon", "return_ann"}
    feats   = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    if not feats:
        raise ValueError("No numeric features detected. Set WINNER_FEATURES in .env or config.")
    return feats


def build_label(df: pd.DataFrame, target_col: str, epsilon: float = 0.0) -> np.ndarray:
    """Binary win label: target_col > epsilon."""
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame.")
    return (pd.to_numeric(df[target_col], errors="coerce") > epsilon).astype(int).values


def build_label_bins4(df: pd.DataFrame, target_col: str, time_col: str,
                      bins_mode: str = "per_day",
                      q: List[float] = None,
                      min_group: int = 20) -> pd.Series:
    """Assign 4-bin quartile labels (0=worst, 3=best).

    bins_mode='per_day':  quartile cut points computed within each trading day
                          (normalises for daily volatility regime).
    bins_mode='global':   cut points computed across the entire dataset.

    Rows with fewer than min_group trades in their group get NaN (excluded from training).
    """
    if q is None:
        q = [0.25, 0.5, 0.75]
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found.")
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found.")

    s = pd.to_numeric(df[target_col], errors="coerce")

    if bins_mode == "global":
        cuts = s.quantile(q).values

        def _to_bin(x):
            if not np.isfinite(x):
                return np.nan
            if x <= cuts[0]: return 0
            if x <= cuts[1]: return 1
            if x <= cuts[2]: return 2
            return 3

        return s.apply(_to_bin).astype("Int64")

    # per_day
    t   = pd.to_datetime(df[time_col], errors="coerce")
    day = t.dt.tz_localize(None).dt.normalize()
    y   = pd.Series(np.nan, index=df.index, dtype="float")

    for _d, idx in day.groupby(day).groups.items():
        idx = list(idx)
        ss  = s.loc[idx]
        if ss.notna().sum() < min_group:
            continue
        cuts = ss.quantile(q).values

        def _to_bin(x, _cuts=cuts):
            if not np.isfinite(x):
                return np.nan
            if x <= _cuts[0]: return 0
            if x <= _cuts[1]: return 1
            if x <= _cuts[2]: return 2
            return 3

        y.loc[idx] = ss.apply(_to_bin)

    return y.astype("Int64")


# ---------------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------------

class DataPreprocessor:
    """Feature imputation, sample weighting, and data prep for training."""

    def __init__(self, cfg: WinnerClassifierConfig):
        self.cfg = cfg
        self.medians_global: Dict[str, float] | None = None

    def compute_medians(self, df: pd.DataFrame, features: List[str]) -> Dict[str, float]:
        return {c: float(pd.to_numeric(df[c], errors="coerce").median()) for c in features}

    def apply_impute(self, df: pd.DataFrame, features: List[str],
                     medians: Dict[str, float]) -> pd.DataFrame:
        X = df[features].copy()
        for c in features:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(medians[c])
        return X

    def drop_na_rows(self, X: pd.DataFrame, y: pd.Series, w: pd.Series | None):
        mask = (~X.isna().any(axis=1)) & (~y.isna())
        return X[mask], y[mask], (w[mask] if w is not None else None)

    def compute_sample_weights(self, df: pd.DataFrame) -> np.ndarray | None:
        if not self.cfg.use_weights:
            return None
        ret     = pd.to_numeric(df[self.cfg.train_target], errors="coerce").fillna(0.0)
        weights = 1.0 + self.cfg.weight_alpha * ret.abs()
        return np.clip(weights, self.cfg.weight_min, self.cfg.weight_max).values

    def prepare(self, df: pd.DataFrame):
        """Run all pre-training transforms; return (df, y, features, weights, has_time)."""
        df = add_dte_and_normalized_returns(df)

        if self.cfg.label_mode == "bins4":
            y_series = build_label_bins4(
                df,
                target_col=self.cfg.train_target,
                time_col=self.cfg.time_col,
                bins_mode=self.cfg.bins_mode,
                q=self.cfg.bins_q,
                min_group=self.cfg.bins_min_group,
            )
            # Drop rows where label is NaN (too few trades in that day)
            mask = y_series.notna()
            df   = df.loc[mask].reset_index(drop=True)
            y    = y_series.loc[mask].astype(int).to_numpy()
        else:
            y = build_label(df, self.cfg.train_target)

        features = select_features(df, self.cfg.features, self.cfg.id_cols)
        weights  = self.compute_sample_weights(df)
        has_time = "trade_date" in df.columns
        if self.cfg.impute_missing:
            self.medians_global = self.compute_medians(df, features)
        return df, y, features, weights, has_time


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

class ModelFactory:
    """Create a classifier for the configured model_type."""

    @staticmethod
    def create(cfg: WinnerClassifierConfig, seed_offset: int = 0):
        seed = cfg.random_state + seed_offset

        is_4bin = (cfg.label_mode == "bins4")

        if cfg.model_type == "lgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators      = int(os.getenv("LGBM_N_ESTIMATORS",      "1000")),
                learning_rate     = float(os.getenv("LGBM_LR",               "0.05")),
                num_leaves        = int(os.getenv("LGBM_NUM_LEAVES",         "42")),
                max_depth         = int(os.getenv("LGBM_MAX_DEPTH",          "-1")),
                min_child_samples = int(os.getenv("LGBM_MIN_CHILD",          "30")),
                min_split_gain    = float(os.getenv("LGBM_MIN_SPLIT_GAIN",   "0.001")),
                subsample         = float(os.getenv("LGBM_BAGGING_FRACTION", "1.0")),
                subsample_freq    = int(os.getenv("LGBM_BAGGING_FREQ",       "0")),
                colsample_bytree  = float(os.getenv("LGBM_FEATURE_FRACTION", "0.8")),
                reg_lambda        = float(os.getenv("LGBM_L2",               "5.0")),
                objective         = "multiclass" if is_4bin else "binary",
                num_class         = 4 if is_4bin else None,
                random_state      = seed,
                n_jobs            = -1,
            )
        elif cfg.model_type == "catboost":
            from catboost import CatBoostClassifier
            return CatBoostClassifier(
                iterations   = int(os.getenv("CAT_ITERS", "4000")),
                learning_rate= float(os.getenv("CAT_LR",   "0.05")),
                depth        = int(os.getenv("CAT_DEPTH",  "6")),
                l2_leaf_reg  = float(os.getenv("CAT_L2",   "6.0")),
                loss_function= "MultiClass" if is_4bin else "Logloss",
                eval_metric  = "MultiClass" if is_4bin else os.getenv("CAT_EVAL_METRIC", "AUC"),
                random_seed  = seed,
                verbose      = False,
                task_type    = os.getenv("CAT_TASK_TYPE", "CPU"),
            )
        else:   # default: RandomForest
            return RandomForestClassifier(
                n_estimators      = cfg.n_estimators,
                max_depth         = cfg.max_depth,
                min_samples_leaf  = cfg.min_samples_leaf,
                min_samples_split = cfg.min_samples_split,
                n_jobs            = -1,
                random_state      = seed,
                class_weight      = cfg.class_weight,
            )


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

class CrossValidator:
    """Out-of-fold (OOF) cross-validation with TimeSeriesSplit or StratifiedKFold."""

    def __init__(self, cfg: WinnerClassifierConfig, preprocessor: DataPreprocessor):
        self.cfg  = cfg
        self.prep = preprocessor

    def _splitter(self, df, y, has_time):
        use_ts = (self.cfg.time_series == "1") or (
            self.cfg.time_series == "auto" and has_time
        )
        if use_ts:
            sp = TimeSeriesSplit(n_splits=self.cfg.oof_folds)
            return sp.split(df), "TimeSeriesSplit"
        sp = StratifiedKFold(n_splits=self.cfg.oof_folds, shuffle=True,
                             random_state=self.cfg.random_state)
        return sp.split(df, y), "StratifiedKFold"

    def _fit_fold(self, clf, Xtr, ytr, wtr, Xva, yva):
        use_es = (self.cfg.model_type in {"lgbm", "catboost"} and
                  self.cfg.valid_fraction > 0 and len(Xva) > 0)
        if use_es:
            cut  = int(len(Xtr) * (1.0 - self.cfg.valid_fraction))
            Xf, yf, wf = Xtr.iloc[:cut], ytr[:cut], wtr[:cut] if wtr is not None else None
            Xe, ye     = Xtr.iloc[cut:], ytr[cut:]
            if self.cfg.model_type == "catboost":
                clf.fit(Xf, yf, sample_weight=wf, eval_set=(Xe, ye),
                        use_best_model=True,
                        early_stopping_rounds=self.cfg.early_stopping_rounds, verbose=False)
            else:  # lgbm
                import lightgbm as lgb
                # bins4 uses multiclass objective → metric must be multi_logloss
                # binary uses binary objective → metric can be aucpr
                lgbm_metric = "multi_logloss" if self.cfg.label_mode == "bins4" else "aucpr"
                clf.fit(Xf, yf, sample_weight=wf, eval_set=[(Xe, ye)],
                        eval_metric=lgbm_metric,
                        callbacks=[lgb.early_stopping(self.cfg.early_stopping_rounds, verbose=False)])
        else:
            clf.fit(Xtr, ytr, sample_weight=wtr)
        return clf

    def run(self, df, y, features, weights, has_time):
        split_iter, split_kind = self._splitter(df, y, has_time)
        is_4bin   = (self.cfg.label_mode == "bins4")
        n_classes = 4 if is_4bin else 2

        # proba_oof shape: (N, 4) for bins4, (N,) for binary
        proba_oof = (np.full((len(df), n_classes), np.nan, dtype=float)
                     if is_4bin else np.full(len(df), np.nan, dtype=float))
        fold_idx  = np.full(len(df), -1, dtype=int)

        for k, (tr, va) in enumerate(split_iter):
            if self.cfg.impute_missing:
                meds = self.prep.compute_medians(df.iloc[tr], features)
                Xtr  = self.prep.apply_impute(df.iloc[tr], features, meds)
                Xva  = self.prep.apply_impute(df.iloc[va], features, meds)
                ytr, yva_k = y[tr], y[va]
                wtr  = None if weights is None else weights[tr]
            else:
                Xtr = df.iloc[tr][features].apply(pd.to_numeric, errors="coerce")
                Xva = df.iloc[va][features].apply(pd.to_numeric, errors="coerce")
                Xtr, _ytr, wtr_s = self.prep.drop_na_rows(
                    Xtr, pd.Series(y[tr]),
                    None if weights is None else pd.Series(weights[tr])
                )
                ytr = _ytr.values
                wtr = None if wtr_s is None else wtr_s.values
                mask_va = ~Xva.isna().any(axis=1)
                Xva     = Xva[mask_va]
                va      = np.asarray(va)[mask_va.values]
                yva_k   = y[va]

            clf  = ModelFactory.create(self.cfg, k)
            clf  = self._fit_fold(clf, Xtr, ytr, wtr, Xva, yva_k)
            pva  = clf.predict_proba(Xva)
            if is_4bin:
                proba_oof[va, :] = pva          # (n_val, 4)
            else:
                proba_oof[va]    = pva[:, 1]    # scalar proba for class=1
            fold_idx[va] = k

        if not is_4bin:
            # Fill any NaN slots with median (binary only)
            nan_mask = np.isnan(proba_oof)
            if nan_mask.any():
                proba_oof[nan_mask] = np.nanmedian(proba_oof)

        return proba_oof, fold_idx, split_kind


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def pick_threshold_by_target(y_true, proba, targets_recall, targets_precision) -> pd.DataFrame:
    """Return a DataFrame of thresholds/metrics at each requested target."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    rows = []

    def _metrics(thr):
        yh = (proba >= thr).astype(int)
        return (precision_score(y_true, yh, zero_division=0),
                recall_score(y_true, yh, zero_division=0),
                f1_score(y_true, yh, zero_division=0),
                float(yh.mean()))

    for tgt in (targets_recall or []):
        chosen_thr, best = None, None
        for thr in sorted(thresholds, reverse=True):
            pr, rc, f1, keep = _metrics(thr)
            if rc >= tgt:
                chosen_thr, best = thr, (pr, rc, f1, keep)
        if chosen_thr is None and len(thresholds):
            chosen_thr = float(min(thresholds))
            best = _metrics(chosen_thr)
        if chosen_thr is not None:
            pr, rc, f1, keep = best
            rows.append(dict(target_type="recall",    target=tgt, threshold=chosen_thr,
                             precision=pr, recall=rc, f1=f1, coverage=keep))

    for tgt in (targets_precision or []):
        chosen_thr, best = None, None
        for i, thr in enumerate(sorted(thresholds)):
            if i % 100 != 0:
                continue
            pr, rc, f1, keep = _metrics(thr)
            if pr >= tgt:
                chosen_thr, best = thr, (pr, rc, f1, keep)
                break
        if chosen_thr is None and len(thresholds):
            chosen_thr = float(max(thresholds))
            best = _metrics(chosen_thr)
        if chosen_thr is not None:
            pr, rc, f1, keep = best
            rows.append(dict(target_type="precision", target=tgt, threshold=chosen_thr,
                             precision=pr, recall=rc, f1=f1, coverage=keep))

    return pd.DataFrame(rows)


def save_feature_importances(clf, X_ref, y_ref, features, out_dir):
    """Save feature importance CSV and PNG (falls back to permutation if needed)."""
    fi_df = None
    if hasattr(clf, "feature_importances_"):
        fi_df = pd.DataFrame({"feature": features,
                               "importance": clf.feature_importances_}).sort_values(
            "importance", ascending=False)
    else:
        try:
            perm  = permutation_importance(clf, X_ref, y_ref, n_repeats=10,
                                           random_state=42, n_jobs=-1)
            fi_df = pd.DataFrame({"feature": features,
                                   "importance": perm.importances_mean,
                                   "importance_std": perm.importances_std}).sort_values(
                "importance", ascending=False)
        except Exception as e:
            print(f"[WARN] Feature importance computation failed: {e}")
            return None

    fi_df.to_csv(os.path.join(out_dir, "winner_feature_importances.csv"), index=False)
    top = fi_df.head(30).iloc[::-1]
    plt.figure(figsize=(8, max(6, 0.3 * len(top))))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.title("Winner Classifier — Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "winner_feature_importances.png"), dpi=150)
    plt.close()
    return fi_df


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_evaluation_outputs(cfg, df, y, proba_oof, fold_idx, split_kind):
    """Write PR curves, threshold table, and OOF scores.

    For bins4 mode the OOF file uses the columns required by b03_train_tail:
      row_idx, y_true, y_pred, p_bin0, p_bin1, p_bin2, p_bin3, fold, has_oof

    For binary mode the OOF file uses the original format:
      row_idx, proba_oof, label, fold
    """
    if cfg.label_mode == "bins4":
        y_pred = np.argmax(proba_oof, axis=1)
        oof_out = pd.DataFrame({
            "row_idx": np.arange(len(df)),
            "y_true":  y,
            "y_pred":  y_pred,
            "p_bin0":  proba_oof[:, 0],
            "p_bin1":  proba_oof[:, 1],
            "p_bin2":  proba_oof[:, 2],
            "p_bin3":  proba_oof[:, 3],
            "fold":    fold_idx,
        })
        oof_out["has_oof"] = (oof_out["fold"] != -1).astype(int)
        oof_out.to_csv(os.path.join(cfg.output_dir, "winner_scores_oof.csv"), index=False)

        # Multi-class accuracy metrics (no PR curve for multiclass)
        valid   = oof_out["has_oof"] == 1
        acc     = float((y_pred[valid] == y[valid]).mean())
        f1m     = float(f1_score(y[valid], y_pred[valid], average="macro", zero_division=0))
        print(f"[INFO] bins4 OOF  Accuracy={acc:.4f}  F1-macro={f1m:.4f}")
        return

    # --- binary mode ---
    precision, recall, thresholds = precision_recall_curve(y, proba_oof)
    coverage = [(proba_oof >= t).mean() for t in thresholds]

    pr_df = pd.DataFrame({"threshold": thresholds,
                           "precision": precision[:-1],
                           "recall":    recall[:-1],
                           "coverage":  coverage})
    pr_df.to_csv(os.path.join(cfg.output_dir, "precision_recall_coverage.csv"), index=False)

    thr_table = pick_threshold_by_target(y, proba_oof, cfg.targets_recall, cfg.targets_precision)
    thr_table.to_csv(os.path.join(cfg.output_dir, "threshold_table.csv"), index=False)

    oof_out = pd.DataFrame({"row_idx": np.arange(len(df)),
                             "proba_oof": proba_oof, "label": y, "fold": fold_idx})
    for c in (cfg.id_cols or []):
        if c in df.columns:
            oof_out[c] = df[c].values
    oof_out.to_csv(os.path.join(cfg.output_dir, "winner_scores_oof.csv"), index=False)

    plt.figure(figsize=(8, 6))
    plt.plot(pr_df["coverage"], pr_df["precision"], label="Precision")
    plt.plot(pr_df["coverage"], pr_df["recall"],    label="Recall")
    plt.xlabel("Coverage (fraction predicted winners)")
    plt.ylabel("Score")
    plt.title(f"PR vs Coverage — OOF ({split_kind}, folds={cfg.oof_folds})")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(cfg.output_dir, "precision_recall_coverage.png"), dpi=150)
    plt.close()


def train_final_model(cfg, prep, df, y, features):
    """Train on all data and return (model, X_all, y_all)."""
    if cfg.impute_missing:
        meds  = prep.medians_global or prep.compute_medians(df, features)
        X_all = prep.apply_impute(df, features, meds)
        y_all = y
    else:
        X_all = df[features].apply(pd.to_numeric, errors="coerce")
        X_all, _y, _ = prep.drop_na_rows(X_all, pd.Series(y), None)
        y_all = _y.values

    clf = ModelFactory.create(cfg)
    clf.fit(X_all, y_all)
    print(f"[INFO] Final model trained on {len(X_all)} rows.")

    try:
        save_feature_importances(clf, X_all, y_all, features, cfg.output_dir)
    except Exception as e:
        print(f"[WARN] Feature importance error: {e}")

    return clf, X_all, y_all


def save_model_pack(cfg, prep, clf, features, df, proba_oof, y, roc_auc, pr_auc, split_kind,
                    fold_idx: np.ndarray = None):
    """Write metrics JSON and save model pack pickle.

    For bins4 mode: multi-class accuracy + top-vs-bottom spread metrics.
                    Metrics are computed only on valid OOF rows (fold_idx != -1).
    For binary mode: standard AUC + best-F1 threshold.
    """
    medians = prep.compute_medians(df, features) if cfg.impute_missing else None

    if cfg.label_mode == "bins4":
        # Only evaluate rows that actually appeared in a validation fold.
        # Rows with fold_idx == -1 (early TimeSeriesSplit rows) have NaN probabilities
        # — including them in metrics would corrupt accuracy/F1.
        if fold_idx is not None:
            valid = (fold_idx != -1)
        else:
            # Fallback: p_bin0 is NaN for unscored rows, NaN >= 0 is False
            valid = proba_oof[:, 0] >= 0
        y_valid = y[valid]
        y_pred  = np.argmax(proba_oof[valid], axis=1)
        acc     = float((y_pred == y_valid).mean())
        f1m     = float(f1_score(y_valid, y_pred, average="macro", zero_division=0))

        # Top-vs-bottom decile spread (key trading metric from CSP_4bin_classifier_strategy_phase_brief.md)
        # Measures whether high p_bin3 scores select better trades than low scores.
        score = proba_oof[valid, 3]
        # df is reset-indexed after prepare(); boolean indexing via numpy is safe
        sret  = pd.to_numeric(
            df[cfg.train_target].to_numpy()[valid], errors="coerce"
        )
        sret  = np.nan_to_num(sret, nan=0.0)
        q90   = np.quantile(score, 0.90)
        q10   = np.quantile(score, 0.10)
        top_mean = float(sret[score >= q90].mean()) if np.any(score >= q90) else float("nan")
        bot_mean = float(sret[score <= q10].mean()) if np.any(score <= q10) else float("nan")
        spread   = float(top_mean - bot_mean) if np.isfinite(top_mean) and np.isfinite(bot_mean) else float("nan")
        print(f"[INFO] bins4 spread  top10%_mean={top_mean:.3f}  bot10%_mean={bot_mean:.3f}  "
              f"spread={spread:.3f}  (target={cfg.train_target})")
        metrics = {
            "task":                   "bins4_multiclass",
            "accuracy":               acc,
            "f1_macro":               f1m,
            # Ranking spread: key signal of trading usefulness (CSP_4bin_classifier_strategy_phase_brief.md §8)
            "top10pct_mean_target":   top_mean,
            "bottom10pct_mean_target": bot_mean,
            "top_minus_bottom_spread": spread,
            "n_oof_valid":            int(valid.sum()),
            "n_rows":                 int(len(df)),
            "n_features":             int(len(features)),
            "features":               features,
            "cv":                     split_kind,
            "bins_mode":              cfg.bins_mode,
            "bins_q":                 cfg.bins_q,
            "bins_min_group":         cfg.bins_min_group,
            "train_target":           cfg.train_target,
        }
        pack = {
            "model":          clf,
            "model_type":     cfg.model_type,
            "label_mode":     "bins4",
            "features":       features,
            "medians":        medians,
            "impute_missing": cfg.impute_missing,
            "metrics":        metrics,
        }
    else:
        precision_arr, recall_arr, thresholds = precision_recall_curve(y, proba_oof)
        best_f1, best_thr = -1.0, 0.5
        for i, thr in enumerate(thresholds):
            if i % 10 != 0 and i != len(thresholds) - 1:
                continue
            yh  = (proba_oof >= thr).astype(int)
            f1  = f1_score(y, yh, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)
        cm = confusion_matrix(y, (proba_oof >= best_thr).astype(int))
        tn, fp, fn, tp = cm.ravel()
        metrics = {
            "roc_auc_oof":              float(roc_auc),
            "pr_auc_oof":               float(pr_auc),
            "best_f1_threshold_oof":    float(best_thr),
            "best_f1_oof":              float(best_f1),
            "coverage_at_best_f1_oof":  float((proba_oof >= best_thr).mean()),
            "confusion_at_best_f1_oof": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "features":                 features,
            "impute_missing":           cfg.impute_missing,
            "use_weights":              cfg.use_weights,
            "cv":                       {"kind": split_kind, "folds": cfg.oof_folds},
        }
        pack = {
            "model":          clf,
            "model_type":     cfg.model_type,
            "label_mode":     "binary",
            "features":       features,
            "medians":        medians,
            "impute_missing": cfg.impute_missing,
            "metrics":        metrics,
            "label":          f"{cfg.train_target} > 0",
        }

    with open(os.path.join(cfg.output_dir, "winner_classifier_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    fname = f"{cfg.model_name}_{cfg.model_type}.pkl"
    joblib.dump(pack, os.path.join(cfg.output_dir, fname))
    print(f"[INFO] Model pack saved → {cfg.output_dir}/{fname}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cfg   = WinnerClassifierConfig()
    prep  = DataPreprocessor(cfg)
    cv    = CrossValidator(cfg, prep)

    # Load and sort data (prefer parquet over stale CSV export)
    from service.table_store import resolve_read_path
    actual_path = resolve_read_path(cfg.input_csv) if table_exists(cfg.input_csv) else cfg.input_csv
    print(f"[INFO] Loading training data: {actual_path}")
    df = read_table(cfg.input_csv) if table_exists(cfg.input_csv) else pd.read_csv(cfg.input_csv)
    df = df.sort_values(["captureTime", "symbol"], kind="mergesort").reset_index(drop=True)
    print(f"[INFO] Loaded {len(df):,} rows")

    # Optionally merge GEX indicator features
    if cfg.gex_folder:
        try:
            from pipeline.b13_train_tail_gex import load_gex_indicators, merge_gex_to_trades
            print(f"[INFO] Loading GEX indicators from: {cfg.gex_folder}")
            gex = load_gex_indicators(cfg.gex_folder)
            df = merge_gex_to_trades(df, gex)
            gex_present = [f for f in NEW_GEX_IND_FEATS if f in df.columns]
            if gex_present:
                cfg.features = cfg.features + gex_present
                match_rate = df["distance_to_flip"].notna().mean()
                print(f"[INFO] Added {len(gex_present)} GEX indicator features "
                      f"(match rate {match_rate:.1%}): {gex_present}")
        except Exception as e:
            print(f"[WARN] GEX merge failed ({e}) — training without GEX indicator features")

    # Use the score date (YYYYMMDD) as the model identifier — self-documenting and
    # consistent with all other output filenames in the rolling-window pipeline.
    score_date     = _cfg_loader.get_score_date()   # e.g. "20251027"
    cfg.output_dir = cfg.output_dir.rstrip("/\\")   # already fully resolved by config template
    cfg.model_name = f"winner_classifier_model_{score_date}"
    ensure_dir(cfg.output_dir)

    # Prepare data (bins4 or binary labeling applied here)
    df, y, features, weights, has_time = prep.prepare(df)

    print(f"[INFO] Training on {len(df)} rows, {len(features)} features, "
          f"model={cfg.model_type}, label_mode={cfg.label_mode}, folds={cfg.oof_folds}")

    # OOF cross-validation
    t0 = pd.Timestamp.now()
    proba_oof, fold_idx, split_kind = cv.run(df, y, features, weights, has_time)
    print(f"[INFO] OOF CV done in {pd.Timestamp.now() - t0}")

    # Mode-specific metrics
    roc_auc = float("nan")
    pr_auc  = float("nan")
    if cfg.label_mode == "bins4":
        valid  = fold_idx != -1
        y_pred = np.argmax(proba_oof[valid], axis=1)
        acc    = float((y_pred == y[valid]).mean())
        f1m    = float(f1_score(y[valid], y_pred, average="macro", zero_division=0))
        print(f"[INFO] bins4 OOF  Accuracy={acc:.4f}  F1-macro={f1m:.4f}")
    else:
        roc_auc = roc_auc_score(y, proba_oof) if len(np.unique(y)) > 1 else float("nan")
        pr_auc  = average_precision_score(y, proba_oof)
        print(f"[INFO] binary OOF  AUC-ROC={roc_auc:.4f}  AUC-PRC={pr_auc:.4f}")

    # Save OOF scores (bins4 format consumed by b03_train_tail)
    save_evaluation_outputs(cfg, df, y, proba_oof, fold_idx, split_kind)

    # Train final model on all data
    clf, X_all, y_all = train_final_model(cfg, prep, df, y, features)

    # Save model pack + metrics (fold_idx needed to filter valid OOF rows in bins4 mode)
    save_model_pack(cfg, prep, clf, features, df, proba_oof, y, roc_auc, pr_auc, split_kind,
                    fold_idx=fold_idx)

    oof_path = os.path.join(cfg.output_dir, "winner_scores_oof.csv")
    print(f"\n✅ Winner classifier trained ({cfg.label_mode} mode).")
    if cfg.label_mode == "bins4":
        print(f"   bins4 OOF Accuracy={acc:.4f}  F1-macro={f1m:.4f}")
        print(f"   OOF scores → {oof_path}  (feed into b03_train_tail)")
    else:
        print(f"   ROC AUC (OOF)={roc_auc:.4f}   PR AUC (OOF)={pr_auc:.4f}")
    print(f"   CV: {split_kind}, folds={cfg.oof_folds}")
    print(f"   Outputs → {cfg.output_dir}/")


if __name__ == "__main__":
    main()
