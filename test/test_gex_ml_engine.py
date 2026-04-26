"""
Smoke tests for the GEX-only ML engine (b13 / b14).

Tests cover:
  - GEX snapshot loading and deduplication (one per symbol/date)
  - Backward asof merge with trades
  - Regime label encoding (train + inference)
  - Full b13 train → b14 score round-trip on synthetic data
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service.constants import NEW_GEX_IND_FEATS
from pipeline.b13_train_tail_gex import (
    build_feat_list,
    encode_regime,
    load_gex_indicators,
    merge_gex_to_trades,
)
from pipeline.b14_score_tail_gex import apply_regime_encoder


# ---------------------------------------------------------------------------
# Synthetic GEX CSV factory
# ---------------------------------------------------------------------------

def make_gex_csv(tmp_path: Path, n_symbols: int = 3, n_days: int = 5) -> Path:
    """Write a single GEX CSV to tmp_path and return its path."""
    import datetime
    rows = []
    for sym_i in range(n_symbols):
        symbol = f"SYM{sym_i}"
        for day in range(n_days):
            dt = datetime.datetime(2025, 10, 1 + day, 11, 0, 0)
            rows.append({
                "symbol":                    symbol,
                "capture_dt":                dt.isoformat(),
                "last_price":                100.0 + sym_i,
                "gamma_flip":                0.5 + sym_i * 0.1,
                "distance_to_flip":          5.0 - day * 0.2,
                "flip_score":                0.7,
                "gamma_density_near":        1.2,
                "gamma_density_below":       0.8,
                "support_asymmetry":         0.3,
                "downside_void":             0.1 * sym_i,
                "peak_concentration":        0.9,
                "slope_near_price":          0.05,
                "negative_gamma_below_ratio": 0.4,
                "has_flip":                  1,
                "no_flip_flag":              0,
                "regime":                    "positive" if sym_i % 2 == 0 else "negative",
            })
    gex_dir = tmp_path / "gex_indicators"
    gex_dir.mkdir()
    csv_path = gex_dir / "indicators_2025-10.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return gex_dir


def make_trades(n_symbols: int = 3, n_per_sym: int = 4) -> pd.DataFrame:
    import datetime
    rows = []
    for sym_i in range(n_symbols):
        for t in range(n_per_sym):
            # tradeTime is after the 11:00 GEX snapshot → should match
            dt = datetime.datetime(2025, 10, 1 + t, 14, 30, 0)
            rows.append({
                "baseSymbol":           f"SYM{sym_i}",
                "tradeTime":            dt.isoformat(),
                "return_mon":           0.01 * (t - n_per_sym // 2),
                "strike":               100,
                "daysToExpiration":     7,
                "potentialReturnAnnual": 0.12,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_new_gex_ind_feats_length():
    assert len(NEW_GEX_IND_FEATS) == 13
    assert "regime" in NEW_GEX_IND_FEATS


def test_build_feat_list_replaces_regime():
    feats = build_feat_list()
    assert "regime" not in feats
    assert "regime_enc" in feats
    assert len(feats) == len(NEW_GEX_IND_FEATS)


def test_load_gex_indicators(tmp_path):
    gex_dir = make_gex_csv(tmp_path, n_symbols=2, n_days=3)
    gex = load_gex_indicators(str(gex_dir))
    # One row per (symbol, date)
    assert len(gex) == 2 * 3
    assert "capture_dt" in gex.columns
    assert "distance_to_flip" in gex.columns


def test_load_gex_indicators_no_files(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_gex_indicators(str(empty_dir))


def test_merge_gex_backward(tmp_path):
    gex_dir = make_gex_csv(tmp_path, n_symbols=3, n_days=5)
    gex     = load_gex_indicators(str(gex_dir))
    trades  = make_trades(n_symbols=3, n_per_sym=4)

    merged = merge_gex_to_trades(trades, gex)

    assert len(merged) == len(trades)
    # Every trade should have a GEX match (tradeTime=14:30 > 11:00 capture_dt)
    assert merged["distance_to_flip"].notna().all()


def test_merge_gex_no_future_leak(tmp_path):
    """GEX snapshots with capture_dt > tradeTime must NOT be merged in."""
    import datetime
    gex_dir = tmp_path / "gex"
    gex_dir.mkdir()
    rows = [{
        "symbol": "AAPL",
        "capture_dt": datetime.datetime(2025, 10, 5, 11, 0).isoformat(),
        "last_price": 100, "gamma_flip": 1, "distance_to_flip": 5,
        "flip_score": 0.5, "gamma_density_near": 1, "gamma_density_below": 1,
        "support_asymmetry": 0, "downside_void": 0, "peak_concentration": 1,
        "slope_near_price": 0, "negative_gamma_below_ratio": 0,
        "has_flip": 1, "no_flip_flag": 0, "regime": "positive",
    }]
    pd.DataFrame(rows).to_csv(gex_dir / "test.csv", index=False)
    gex = load_gex_indicators(str(gex_dir))

    # Trade happens BEFORE the GEX snapshot
    trade = pd.DataFrame([{
        "baseSymbol": "AAPL",
        "tradeTime":  datetime.datetime(2025, 10, 3, 14, 30).isoformat(),
        "return_mon": 0.01,
    }])
    merged = merge_gex_to_trades(trade, gex)
    assert pd.isna(merged["distance_to_flip"].iloc[0])


def test_encode_regime_train():
    df = pd.DataFrame({"regime": ["positive", "negative", "neutral", None]})
    df_enc, le = encode_regime(df)
    assert "regime_enc" in df_enc.columns
    assert df_enc["regime_enc"].notna().all()
    assert "unknown" in le.classes_


def test_encode_regime_inference_unknown():
    df_train = pd.DataFrame({"regime": ["positive", "negative"]})
    _, le = encode_regime(df_train)

    df_score = pd.DataFrame({"regime": ["positive", "NEW_REGIME"]})
    df_score = apply_regime_encoder(df_score, le)
    # "NEW_REGIME" → "unknown" → valid encoded int
    assert "regime_enc" in df_score.columns
    assert df_score["regime_enc"].notna().all()


def test_b13_train_b14_score_roundtrip(tmp_path, monkeypatch):
    """Full round-trip: synthetic data → b13 train → b14 score."""
    import os
    import joblib
    from service.tail_scoring import fill_features
    from pipeline.b13_train_tail_gex import (
        GexTailConfig, build_feat_list, encode_regime, run_oof,
        find_best_f1_threshold,
    )
    from pipeline.b14_score_tail_gex import apply_regime_encoder

    # --- Build synthetic inputs ---
    gex_dir = make_gex_csv(tmp_path, n_symbols=3, n_days=20)
    gex     = load_gex_indicators(str(gex_dir))
    trades  = make_trades(n_symbols=3, n_per_sym=20)

    df = merge_gex_to_trades(trades, gex)
    df, le = encode_regime(df)

    feat_list = build_feat_list()
    # Tail label: worst 20% (need enough positives with small data)
    tail_cut = float(df["return_mon"].quantile(0.20))
    y = (df["return_mon"] <= tail_cut).astype(int).values

    df = df.sort_values("tradeTime").reset_index(drop=True)
    y  = (df["return_mon"] <= tail_cut).astype(int).values
    X, medians = fill_features(df, feat_list)

    assert X.shape[1] == len(feat_list)

    # --- Train with 2 folds (small data) ---
    from sklearn.isotonic import IsotonicRegression
    from lightgbm import LGBMClassifier

    oof_proba, _ = run_oof(df, X, y, n_splits=2)
    valid_mask = oof_proba > 0

    cal = IsotonicRegression(out_of_bounds="clip")
    if valid_mask.sum() > 1:
        cal.fit(oof_proba[valid_mask], y[valid_mask])

    pos = y.sum(); neg = len(y) - pos
    model = LGBMClassifier(
        n_estimators=10, learning_rate=0.1, num_leaves=4,
        scale_pos_weight=neg / max(pos, 1), random_state=42, verbose=-1,
    )
    model.fit(X, y)

    pack_path = str(tmp_path / "tail_gex_model_test.pkl")
    joblib.dump({
        "model": model, "calibrator": cal, "features": feat_list,
        "medians": medians, "regime_encoder": le,
        "oof_best_threshold": 0.3, "oof_auc": 0.6,
        "oof_avg_precision": 0.2, "tail_rate": float(y.mean()),
        "contamination_rate": 0.0,
    }, pack_path)

    # --- Score with b14 logic ---
    pack = joblib.load(pack_path)
    score_df = trades.copy()
    score_df = merge_gex_to_trades(score_df, gex)
    score_df = apply_regime_encoder(score_df, pack["regime_encoder"])

    X_score, _ = fill_features(score_df, pack["features"], medians=pack["medians"])
    proba = pack["model"].predict_proba(X_score)[:, 1]
    assert len(proba) == len(score_df)
    assert (proba >= 0).all() and (proba <= 1).all()
