import pandas as pd

from pipeline import a03_filter_trades, a04_label_data, b02_score_winner


def test_a03_load_config_uses_enriched_dataset_path(tmp_path, monkeypatch):
    corp_cfg = tmp_path / "corp_action_config.yaml"
    corp_cfg.write_text(
        """
exclusion_windows:
  earnings:
    days_before_trade: 2
    days_after_trade: 1
    days_before_expiry: 7
    days_after_expiry: 1
""".strip(),
        encoding="utf-8",
    )

    dataset_cfg = {
        "data_basic_csv": "trades_raw_current.csv",
        "events_output": str(tmp_path / "events.csv"),
        "filtered_trades_csv": str(tmp_path / "filtered.csv"),
        "filtered_out_csv": str(tmp_path / "excluded.csv"),
    }

    monkeypatch.setattr(
        "service.env_config.config.get_active_dataset_config",
        lambda: dataset_cfg,
    )
    monkeypatch.setattr(
        "service.env_config.getenv",
        lambda key, default=None: str(tmp_path / "output")
        if key == "COMMON_OUTPUT_DIR"
        else default,
    )

    cfg = a03_filter_trades.load_config(str(corp_cfg))

    assert cfg.trades_csv == str(
        tmp_path / "output" / "data_prep" / "trades_with_gex_macro_current.csv"
    )
    assert cfg.events_csv == str(tmp_path / "events.csv")
    assert cfg.output_csv == str(tmp_path / "filtered.csv")


def test_a04_label_single_dataset_prefers_filtered_csv(tmp_path, monkeypatch):
    filtered_csv = tmp_path / "filtered.csv"
    pd.DataFrame(
        {
            "baseSymbol": ["FILTERED"],
            "expirationDate": ["2025-01-10"],
            "strike": [100],
            "tradeTime": ["2025-01-06 11:00:00"],
            "underlyingLastPrice": [101.0],
            "daysToExpiration": [4],
            "bidPrice": [1.25],
        }
    ).to_csv(filtered_csv, index=False)

    dataset_cfg = {
        "cutoff_date": "2025-01-15",
        "output_csv": "labeled.csv",
        "filtered_trades_csv": str(filtered_csv),
        "data_basic_csv": "trades_raw_current.csv",
    }

    captured = {}

    monkeypatch.setattr(
        "service.env_config.config.get_active_dataset_config",
        lambda: dataset_cfg,
    )
    monkeypatch.setattr(a04_label_data, "load_exclude_symbols", lambda: set())

    def fake_label_csv_file(df, output_csv, cutoff_date):
        captured["df"] = df.copy()
        captured["output_csv"] = output_csv
        captured["cutoff_date"] = cutoff_date

    monkeypatch.setattr(a04_label_data, "label_csv_file", fake_label_csv_file)

    a04_label_data.label_single_dataset()

    assert captured["df"]["baseSymbol"].tolist() == ["FILTERED"]
    assert captured["output_csv"] == "labeled.csv"
    assert captured["cutoff_date"] == "2025-01-15"


def _scoring_config(csv_in: str, split_file: str = "", use_oof: bool = True):
    return b02_score_winner.ScoringConfig(
        csv_in=csv_in,
        model_in="unused.pkl",
        csv_out_dir="unused",
        csv_out="unused.csv",
        proba_col="win_proba",
        pred_col="win_predict",
        train_target="return_mon",
        gex_filter=False,
        model_type="lgbm",
        fixed_threshold=None,
        use_pack_f1=True,
        target_precisions=[],
        target_recalls=[],
        auto_calibrate=False,
        split_file=split_file,
        use_oof=use_oof,
        train_epsilon=0.0,
        write_sweep=False,
        process_on_fly=False,
    )


def test_b02_load_and_preprocess_data_accepts_base_symbol_schema(tmp_path):
    csv_in = tmp_path / "labeled_scores_input.csv"
    pd.DataFrame(
        {
            "baseSymbol": ["SPY", "QQQ"],
            "tradeTime": ["2025-01-06 11:00:00", "2025-01-07 11:00:00"],
            "expirationDate": ["2025-01-10", "2025-01-10"],
            "daysToExpiration": [4, 3],
            "return_pct": [2.0, -1.0],
            "is_train": [0, 1],
        }
    ).to_csv(csv_in, index=False)

    out = b02_score_winner.load_and_preprocess_data(_scoring_config(str(csv_in)))

    assert out["baseSymbol"].tolist() == ["SPY"]
    assert "return_mon" in out.columns


def test_b02_split_file_branch_supports_base_symbol_join(tmp_path):
    csv_in = tmp_path / "labeled_scores_input.csv"
    split_file = tmp_path / "winner_split.csv"

    pd.DataFrame(
        {
            "baseSymbol": ["SPY", "QQQ"],
            "tradeTime": ["2025-01-06 11:00:00", "2025-01-07 11:00:00"],
            "expirationDate": ["2025-01-10", "2025-01-10"],
            "daysToExpiration": [4, 3],
            "return_pct": [2.0, -1.0],
        }
    ).to_csv(csv_in, index=False)

    pd.DataFrame(
        {
            "baseSymbol": ["SPY", "QQQ"],
            "tradeTime": ["2025-01-06 11:00:00", "2025-01-07 11:00:00"],
            "is_train": [0, 1],
        }
    ).to_csv(split_file, index=False)

    out = b02_score_winner.load_and_preprocess_data(
        _scoring_config(str(csv_in), split_file=str(split_file), use_oof=False)
    )

    assert out["baseSymbol"].tolist() == ["SPY"]
