import numpy as np
import pandas as pd

from service.data_prepare import (
    _per_symbol_feature_frame,
    derive_capital,
    lookup_close_on_or_before,
)


def test_lookup_close_on_or_before_respects_max_days_back():
    price_df = pd.DataFrame(
        {"Close": [100.0, 102.5]},
        index=pd.to_datetime(["2025-01-02", "2025-01-06"]),
    )

    assert np.isnan(
        lookup_close_on_or_before(price_df, pd.Timestamp("2025-01-08"), max_days_back=1)
    )
    assert (
        lookup_close_on_or_before(price_df, pd.Timestamp("2025-01-08"), max_days_back=3)
        == 102.5
    )


def test_derive_capital_defaults_to_strike100():
    df = pd.DataFrame(
        {
            "strike": [50.0, 120.0],
            "underlyingLastPrice": [55.0, 130.0],
            "bidPrice": [1.0, 2.0],
            "entry_credit": [100.0, 200.0],
        }
    )

    out = derive_capital(df)

    assert out.tolist() == [5000.0, 12000.0]


def test_per_symbol_feature_frame_does_not_use_future_close_on_holidays():
    prices = pd.Series(
        [100.0, 110.0],
        index=pd.to_datetime(["2024-12-31", "2025-01-02"]),
        name="Close",
    )

    out = _per_symbol_feature_frame(
        prices,
        start_date=pd.Timestamp("2024-12-31"),
        max_trade_date=pd.Timestamp("2025-01-02"),
    )

    # 2025-01-01 is a holiday but appears in the business-day calendar.
    # The 2025-01-02 prev_close must still come from the prior real close.
    assert out.loc[pd.Timestamp("2025-01-02"), "prev_close"] == 100.0
    assert out.loc[pd.Timestamp("2025-01-02"), "prev_close"] != 110.0
