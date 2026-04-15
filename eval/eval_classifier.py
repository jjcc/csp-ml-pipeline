#!/usr/bin/env python3
"""
eval_classifier.py  (v2)

A flexible evaluation script (env-driven) for scored CSVs.
Computes AUC/PR AUC, precision/recall/coverage, confusion matrices, and threshold tables.
Supports label definitions for:
  - Winner:          y = (return_pct > 0)
  - Tail by return:  worst K% by return_pct
  - Tail by PnL:     worst K% by total_pnl
  - Provided:        y from an explicit column

Inputs (via .env):
  EVAL_INPUT=./scores.csv                # CSV with at least a probability column
  EVAL_OUTPUT_DIR=./eval_out

  # How to read probabilities and choose thresholds
  EVAL_PROBA_COL=prob_winner             # name of probability column in the CSV
  EVAL_FIXED_THRESHOLDS=0.5,0.7          # optional fixed thresholds (comma or JSON list)
  EVAL_TARGET_PRECISION=0.90,0.95        # optional precision targets for threshold search
  EVAL_TARGET_RECALL=0.83,0.90           # optional recall targets for threshold search

  # Label construction
  EVAL_LABEL_MODE=winner                 # winner | tail_pct | tail_pnl | provided
  EVAL_TAIL_K=0.05                       # fraction for tail_pct / tail_pnl
  EVAL_LABEL_COL=                        # required if mode=provided (0/1 or boolean)
  EVAL_RETURN_COL=return_pct             # used by winner/tail_pct
  EVAL_PNL_COL=total_pnl                 # used by tail_pnl

  # Optional filtering/grouping
  EVAL_FILTER_QUERY=                     # pandas query string to filter rows pre-eval (e.g., "dte<=7 and moneyness<0.05")
  EVAL_GROUP_COLS=                       # optional group-by columns (comma/JSON); per-group key metrics are emitted

Outputs:
  - metrics.json                   (ROC AUC, PR AUC, base positive rate, etc.)
  - thresholds_table.csv           (rows for fixed thresholds + target-based thresholds)
  - precision_recall_coverage.csv  (curve for plotting)
  - confusion_at_thresholds.csv    (confusion matrices for each chosen threshold)
  - group_metrics.csv              (optional, if EVAL_GROUP_COLS set)
(v2)
Adds a unified "recall_targets_table.csv" built from the PR curve:
columns: target_type, target, threshold, precision, recall, coverage, keep_rate

- coverage = fraction predicted positive (y_hat==1)
- keep_rate = 1 - coverage  (for tail_* modes by default), or = coverage for winner/provided unless overridden

Env (extends previous):
  EVAL_KEEP_RATE_DEF=auto   # auto | positive | negative
    - auto: for tail_* -> negative (1-coverage), else -> positive (coverage)
    - positive: keep_rate := coverage
    - negative: keep_rate := 1 - coverage

Adds optional split-aware evaluation:
- If EVAL_SPLIT_COL is set and present in the CSV, this script will run the same
  evaluation twice: once for overall, and once per value of the split column,
  writing results under:
    <EVAL_OUTPUT_DIR>/split_<SPLITCOL>=<value>/

Everything else (labels, PR export, recall-target table, confusion matrices,
metrics.json, PR plot) is unchanged from your stable version.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os, json
from typing import List, Tuple
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt

def _parse_list(val: str, cast=float) -> List:
    if val is None: return []
    s = val.strip()
    if not s: return []
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            return [cast(x) for x in arr]
    except Exception:
        pass
    return [cast(x.strip()) for x in s.split(",") if x.strip()]

def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def build_labels(df: pd.DataFrame, mode: str, tail_k: float, return_col: str, pnl_col: str, provided_col: str):
    mode = (mode or "winner").lower()
    if mode == "winner":
        ret = pd.to_numeric(df[return_col], errors="coerce")
        y = (ret > 0).astype(int).values
        base_rate = float((ret > 0).mean())
        label_info = {"mode": "winner", "return_col": return_col}
    elif mode == "tail_pct":
        ret = pd.to_numeric(df[return_col], errors="coerce")
        cut = ret.quantile(tail_k)
        y = (ret <= cut).astype(int).values
        base_rate = float((ret <= cut).mean())
        label_info = {"mode": "tail_pct", "return_col": return_col, "k": tail_k, "cut": float(cut)}
    elif mode == "tail_pnl":
        pnl = pd.to_numeric(df[pnl_col], errors="coerce")
        cut = pnl.quantile(tail_k)
        y = (pnl <= cut).astype(int).values
        base_rate = float((pnl <= cut).mean())
        label_info = {"mode": "tail_pnl", "pnl_col": pnl_col, "k": tail_k, "cut": float(cut)}
    elif mode == "provided":
        if not provided_col or provided_col not in df.columns:
            raise SystemExit("EVAL_LABEL_MODE=provided requires EVAL_LABEL_COL to exist in the CSV.")
        y = pd.to_numeric(df[provided_col], errors="coerce").fillna(0).astype(int).values
        base_rate = float((y == 1).mean())
        label_info = {"mode": "provided", "label_col": provided_col}
    else:
        raise SystemExit(f"Unknown EVAL_LABEL_MODE={mode}")
    return y, base_rate, label_info

def metrics_at_threshold(y_true, y_score, thr):
    y_pred = (y_score >= thr).astype(int)
    pr = precision_score(y_true, y_pred, zero_division=0)
    rc = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cov = float(y_pred.mean())
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return dict(
        threshold=float(thr),
        precision=float(pr),
        recall=float(rc),
        f1=float(f1),
        coverage=float(cov),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)
    )

def choose_keep_rate_def(mode: str, setting: str) -> str:
    setting = (setting or "auto").lower()
    if setting in {"positive","negative"}:
        return setting
    # auto
    if mode.startswith("tail"):
        return "negative"  # keep = non-positives
    else:
        return "positive"  # keep = predicted positives

def pick_thresholds_from_targets(y_true, y_score, targets_prec, targets_rec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prec, rec, thr = precision_recall_curve(y_true, y_score)

    rows = []
    # Precision targets: choose lowest threshold achieving >= target precision
    for t in targets_prec:
        idxs = np.where(prec >= t)[0]
        if len(idxs) == 0:
            idx = len(prec) - 1
        else:
            idx = idxs[0]
        chosen_thr = thr[idx] if idx < len(thr) else thr[-1]
        m = metrics_at_threshold(y_true, y_score, chosen_thr)
        m.update({"target_type": "precision", "target": float(t)})
        rows.append(m)

    # Recall targets: choose highest threshold achieving >= target recall
    for t in targets_rec:
        chosen_thr, chosen_m = None, None
        for i in range(len(thr)-1, -1, -1):  # iterate thresholds descending
            m = metrics_at_threshold(y_true, y_score, thr[i])
            if m["recall"] >= t:
                chosen_thr, chosen_m = thr[i], m
        if chosen_m is None and len(thr):
            chosen_thr = float(thr[0])
            chosen_m = metrics_at_threshold(y_true, y_score, chosen_thr)
        chosen_m.update({"target_type": "recall", "target": float(t)})
        rows.append(chosen_m)

    # PR coverage export
    pr_curve = pd.DataFrame({"threshold": thr, "precision": prec[:-1], "recall": rec[:-1]})
    return pd.DataFrame(rows), pr_curve

def build_recall_targets_table(pr_curve: pd.DataFrame, y_true: np.ndarray, y_score: np.ndarray, recall_targets: List[float], keep_def: str) -> pd.DataFrame:
    """Construct a single table that looks like the user's desired output, using PR curve + exact metrics at chosen thr."""
    out = []
    for t in recall_targets:
        # Closest recall
        idx = int((pr_curve["recall"] - t).abs().idxmin())
        thr = float(pr_curve.loc[idx, "threshold"])
        m = metrics_at_threshold(y_true, y_score, thr)
        cov = m["coverage"]
        keep = (1.0 - cov) if keep_def == "negative" else cov
        out.append({
            "target_type": "recall",
            "target": float(t),
            "threshold": thr,
            "precision": m["precision"],
            "recall": m["recall"],
            "coverage": cov,
            "keep_rate": keep
        })
    return pd.DataFrame(out)

def _evaluate_block(df: pd.DataFrame, OUTDIR: str, PROBA_COL: str,
                    MODE: str, TAIL_K: float, LABEL_COL: str,
                    RETURN_COL: str, PNL_COL: str,
                    FIXED_THR: List[float], TGT_PREC: List[float], TGT_RECALL: List[float],
                    KEEP_DEF: str):
    """This is your original evaluation body, extracted so we can call it for overall + each split."""
    ensure_dir(OUTDIR)

    if PROBA_COL not in df.columns:
        raise SystemExit(f"Probability column '{PROBA_COL}' not in CSV.")

    y_score = pd.to_numeric(df[PROBA_COL], errors="coerce").fillna(0.0).values
    y_true, base_rate, label_info = build_labels(df, MODE, TAIL_K, RETURN_COL, PNL_COL, LABEL_COL)

    # AUCs
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc  = average_precision_score(y_true, y_score)

    # Thresholds from targets + PR curve
    tgt_table, pr_curve = pick_thresholds_from_targets(y_true, y_score, TGT_PREC, TGT_RECALL)

    # Fixed thresholds
    fixed_rows = [metrics_at_threshold(y_true, y_score, thr) for thr in FIXED_THR] if FIXED_THR else []
    fixed_table = pd.DataFrame(fixed_rows) if fixed_rows else pd.DataFrame(columns=["threshold","precision","recall","f1","coverage","tn","fp","fn","tp"])

    # Confusions at chosen thresholds (targets + fixed)
    all_rows = []
    if not tgt_table.empty:
        all_rows.append(tgt_table)
    if not fixed_table.empty:
        all_rows.append(fixed_table)
    thresholds_table = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    # Confusions at the chosen thresholds
    conf_rows = []
    for _, r in thresholds_table.iterrows():
        conf_rows.append(dict(kind=r.get("target_type","fixed"), target=r.get("target", None), **metrics_at_threshold(y_true, y_score, float(r["threshold"]))))
    conf_table = pd.DataFrame(conf_rows)

    # Unified recall-targets table (built from PR curve)
    keep_def = choose_keep_rate_def(MODE.lower(), KEEP_DEF)
    recall_targets = TGT_RECALL if TGT_RECALL else []
    unified_table = build_recall_targets_table(pr_curve, y_true, y_score, recall_targets, keep_def) if recall_targets else pd.DataFrame()

    # Save artifacts
    thresholds_path = os.path.join(OUTDIR, "thresholds_table.csv")
    prcov_path      = os.path.join(OUTDIR, "precision_recall_coverage.csv")
    conf_path       = os.path.join(OUTDIR, "confusion_at_thresholds.csv")
    metrics_path    = os.path.join(OUTDIR, "metrics.json")
    unified_path    = os.path.join(OUTDIR, "recall_targets_table.csv")

    thresholds_table.to_csv(thresholds_path, index=False)
    pr_curve.to_csv(prcov_path, index=False)
    conf_table.to_csv(conf_path, index=False)
    if not unified_table.empty:
        unified_table.to_csv(unified_path, index=False)

    metrics = {
        "rows": int(len(df)),
        "mode": MODE,
        "label_info": label_info,
        "base_positive_rate": float(base_rate),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "proba_col": PROBA_COL,
        "fixed_thresholds": FIXED_THR,
        "targets_precision": TGT_PREC,
        "targets_recall": TGT_RECALL,
        "keep_rate_def": keep_def,
        "artifacts": {
            "thresholds_table": thresholds_path,
            "pr_curve": prcov_path,
            "confusions": conf_path,
            "recall_targets_table": unified_path if recall_targets else None
        }
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


    # Plot PR curve for quick glance
    try:
        plt.figure(figsize=(6,5))
        plt.plot(pr_curve["recall"], pr_curve["precision"], label="PR curve")
        if not unified_table.empty:
            for _, r in unified_table.iterrows():
                plt.scatter(r["recall"], r["precision"])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "pr_curve.png"), dpi=150)
        plt.close()
    except Exception as e:
        print("Plotting failed:", e)

    print(f"[OK] Evaluated {len(df)} rows. AUC={roc_auc:.4f}, PR AUC={pr_auc:.4f}")
    if recall_targets:
        print(f"- recall_targets_table.csv: {unified_path}")
    print(f"- thresholds_table.csv: {thresholds_path}")
    print(f"- precision_recall_coverage.csv: {prcov_path}")
    print(f"- confusion_at_thresholds.csv: {conf_path}")
    print(f"- metrics.json: {metrics_path}")

def main():
    load_dotenv()

    # IO
    CSV_IN = os.getenv("EVAL_INPUT", "./scores.csv")
    OUTDIR = os.getenv("EVAL_OUTPUT_DIR", "./eval_out")
    ensure_dir(OUTDIR)

    # Proba + thresholds
    PROBA_COL = os.getenv("EVAL_PROBA_COL", "prob")
    FIXED_THR = _parse_list(os.getenv("EVAL_FIXED_THRESHOLDS", ""), cast=float)
    TGT_PREC  = _parse_list(os.getenv("EVAL_TARGET_PRECISION", ""), cast=float)
    TGT_RECALL= _parse_list(os.getenv("EVAL_TARGET_RECALL", ""), cast=float)

    # Labels
    MODE      = os.getenv("EVAL_LABEL_MODE", "winner")
    TAIL_K    = float(os.getenv("EVAL_TAIL_K", "0.05"))
    LABEL_COL = os.getenv("EVAL_LABEL_COL", "")
    RETURN_COL= os.getenv("EVAL_RETURN_COL", "return_pct")
    PNL_COL   = os.getenv("EVAL_PNL_COL", "total_pnl")

    # Keep rate convention
    KEEP_DEF  = os.getenv("EVAL_KEEP_RATE_DEF", "auto")

    # Optional filtering/grouping
    FILTER_Q  = os.getenv("EVAL_FILTER_QUERY", "").strip()
    GROUP_COLS= _parse_list(os.getenv("EVAL_GROUP_COLS", ""), cast=str)

    # NEW: optional split column
    SPLIT_COL = os.getenv("EVAL_SPLIT_COL", "is_train").strip()

    # Load
    df = pd.read_csv(CSV_IN)

    # Filter if needed
    if FILTER_Q:
        df = df.query(FILTER_Q)

    # ---- Overall (existing behavior) ----
    _evaluate_block(
        df=df, OUTDIR=OUTDIR, PROBA_COL=PROBA_COL,
        MODE=MODE, TAIL_K=TAIL_K, LABEL_COL=LABEL_COL,
        RETURN_COL=RETURN_COL, PNL_COL=PNL_COL,
        FIXED_THR=FIXED_THR, TGT_PREC=TGT_PREC, TGT_RECALL=TGT_RECALL,
        KEEP_DEF=KEEP_DEF
    )

    # ---- Per-split (non-breaking; runs only if SPLIT_COL present) ----
    if SPLIT_COL and SPLIT_COL in df.columns:
        for val, g in df.groupby(SPLIT_COL):
            subdir = os.path.join(OUTDIR, f"split_{SPLIT_COL}={val}")
            _evaluate_block(
                df=g, OUTDIR=subdir, PROBA_COL=PROBA_COL,
                MODE=MODE, TAIL_K=TAIL_K, LABEL_COL=LABEL_COL,
                RETURN_COL=RETURN_COL, PNL_COL=PNL_COL,
                FIXED_THR=FIXED_THR, TGT_PREC=TGT_PREC, TGT_RECALL=TGT_RECALL,
                KEEP_DEF=KEEP_DEF
            )

    # ---- Optional: group summaries (unchanged) ----
    if GROUP_COLS:
        grp_rows = []
        for key, g in df.groupby(GROUP_COLS):
            y_g, _, _ = build_labels(g, MODE, TAIL_K, RETURN_COL, PNL_COL, LABEL_COL)
            s_g = pd.to_numeric(g[PROBA_COL], errors="coerce").fillna(0.0).values
            roc_g = roc_auc_score(y_g, s_g)
            pr_g  = average_precision_score(y_g, s_g)
            grp_rows.append({"group": key if isinstance(key, tuple) else (key,),
                             "rows": int(len(g)), "roc_auc": float(roc_g),
                             "pr_auc": float(pr_g), "base_positive_rate": float((np.array(y_g)==1).mean())})
        pd.DataFrame(grp_rows).to_csv(os.path.join(OUTDIR, "group_metrics.csv"), index=False)

if __name__ == "__main__":
    main()
