from pathlib import Path

import pandas as pd

from service.preprocess import (
    add_dte_and_normalized_returns,
    keep_one_row_per_contract_per_day,
    merge_gex,
    pick_daily_snapshot_files,
)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_pick_daily_snapshot_files_prefers_closest_after_then_latest_before(tmp_path):
    for name in [
        "coveredPut_2025-01-02_10_30.csv",
        "coveredPut_2025-01-02_11_05.csv",
        "coveredPut_2025-01-02_11_20.csv",
        "coveredPut_2025-01-03_09_30.csv",
        "coveredPut_2025-01-03_10_45.csv",
    ]:
        (tmp_path / name).write_text("symbol\nSPY\n", encoding="utf-8")

    picked = pick_daily_snapshot_files(str(tmp_path), "coveredPut_*.csv", "11:00")

    assert [Path(p).name for p in picked] == [
        "coveredPut_2025-01-02_11_05.csv",
        "coveredPut_2025-01-03_10_45.csv",
    ]


def test_keep_one_row_per_contract_per_day_keeps_nearest_target_time():
    df = pd.DataFrame(
        {
            "symbol": ["SPY|1", "SPY|1", "QQQ|1"],
            "captureTime": pd.to_datetime(
                ["2025-01-02 10:50", "2025-01-02 11:03", "2025-01-02 11:30"]
            ),
            "value": [1, 2, 3],
        }
    )

    out = keep_one_row_per_contract_per_day(df, target_time="11:00")

    assert len(out) == 2
    assert out.loc[out["symbol"] == "SPY|1", "value"].item() == 2


def test_merge_gex_adds_feature_columns_and_selected_file(tmp_path):
    gex_file = tmp_path / "2025-01-02" / "spy_2025-01-02_11-00.csv"
    _write_csv(
        gex_file,
        pd.DataFrame(
            {
                "strike": [95, 100, 105],
                "gamma": [-2.0, 1.0, 3.0],
            }
        ),
    )

    trades = pd.DataFrame(
        {
            "baseSymbol": ["SPY"],
            "tradeTime": ["2025-01-02 11:00:00"],
            "underlyingLastPrice": [100.0],
        }
    )

    out = merge_gex(trades, str(tmp_path), target_minutes=11 * 60)

    assert out.loc[0, "gex_missing"] == 0
    assert out.loc[0, "gex_file"].endswith("spy_2025-01-02_11-00.csv")
    assert out.loc[0, "gex_total"] == 2.0
    assert out.loc[0, "gex_pos"] == 4.0
    assert out.loc[0, "gex_neg"] == -2.0


def test_add_dte_and_normalized_returns_computes_return_columns():
    df = pd.DataFrame(
        {
            "tradeTime": ["2025-01-02 11:00:00"],
            "expirationDate": ["2025-01-06"],
            "daysToExpiration": [4],
            "return_pct": [8.0],
        }
    )

    out = add_dte_and_normalized_returns(df)

    assert out.loc[0, "return_per_day"] == 2.0
    assert out.loc[0, "return_ann"] == 8.0 * 365.0 / 4.0
    assert out.loc[0, "return_mon"] == 8.0 * 30.0 / 4.0
