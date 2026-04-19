import joblib
import numpy as np
import pandas as pd

from service.tail_scoring import TailModelPack, add_bin_prob_features, score_tail_data
from service.winner_scoring import (
    WinnerModelPack,
    load_winner_model,
    score_winner_data,
    select_winner_threshold,
)


class DummyBinaryModel:
    def predict_proba(self, X):
        n = len(X)
        positive = np.linspace(0.2, 0.8, n)
        return np.column_stack([1.0 - positive, positive])


class DummyMulticlassModel:
    def predict_proba(self, X):
        n = len(X)
        row = np.array([0.1, 0.2, 0.3, 0.4])
        return np.tile(row, (n, 1))


class DummyTailModel:
    def predict_proba(self, X):
        raw = np.full(len(X), 0.25)
        return np.column_stack([1.0 - raw, raw])


class OffsetCalibrator:
    def transform(self, values):
        return np.clip(np.asarray(values) + 0.1, 0.0, 1.0)


def test_load_winner_model_uses_saved_oof_threshold(tmp_path):
    model_path = tmp_path / "winner.pkl"
    joblib.dump(
        {
            "model": DummyBinaryModel(),
            "features": ["feature_a"],
            "medians": {"feature_a": 0.0},
            "label_mode": "binary",
            "metrics": {"best_f1_threshold_oof": 0.73},
        },
        model_path,
    )

    pack = load_winner_model(str(model_path))

    assert pack.best_f1_threshold == 0.73
    assert pack.label_mode == "binary"


def test_score_winner_data_bins4_uses_p_bin3_as_primary_score():
    df = pd.DataFrame({"feature_a": [1.0, 2.0]})
    pack = WinnerModelPack(
        model=DummyMulticlassModel(),
        features=["feature_a"],
        medians={"feature_a": 0.0},
        impute_missing=True,
        best_f1_threshold=0.5,
        label_mode="bins4",
    )

    scored, proba, _ = score_winner_data(df, pack, proba_col="win_proba")

    assert np.allclose(proba, [0.4, 0.4])
    assert np.allclose(scored["p_bin0"], [0.1, 0.1])
    assert np.allclose(scored["p_bin3"], [0.4, 0.4])
    assert np.allclose(scored["conflict_score"], [0.04, 0.04])
    assert np.allclose(scored["win_proba"], [0.4, 0.4])


def test_select_winner_threshold_prefers_pack_threshold_when_requested():
    chosen, table = select_winner_threshold(
        np.array([0.2, 0.8]),
        np.array([0, 1]),
        use_pack_f1=True,
        best_f1_threshold=0.61,
        auto_calibrate=False,
    )

    assert chosen == 0.61
    assert table is None


def test_add_bin_prob_features_injects_winner_probabilities():
    df = pd.DataFrame({"feature_a": [1.0, 2.0]})
    winner_pack = {
        "model": DummyMulticlassModel(),
        "features": ["feature_a"],
        "medians": {"feature_a": 0.0},
    }

    out = add_bin_prob_features(df, winner_pack)

    assert np.allclose(out["p_bin0"], [0.1, 0.1])
    assert np.allclose(out["p_bin3"], [0.4, 0.4])
    assert np.allclose(out["conflict_score"], [0.04, 0.04])


def test_score_tail_data_applies_calibrator():
    df = pd.DataFrame({"tail_feature": [1.0, 2.0]})
    pack = TailModelPack(
        model=DummyTailModel(),
        calibrator=OffsetCalibrator(),
        features=["tail_feature"],
        medians={"tail_feature": 0.0},
        oof_best_threshold=0.4,
    )

    out, proba = score_tail_data(df, pack)

    assert np.allclose(proba, [0.35, 0.35])
    assert np.allclose(out["tail_proba"], [0.35, 0.35])
