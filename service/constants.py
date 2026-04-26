"""
constants.py — Single source of truth for feature lists and shared constants.

All feature group definitions and magic numbers live here.
Import from this module instead of defining them inline.
"""

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

# Core option greeks and market microstructure
BASE_FEATS = [
    "breakEvenProbability",
    "moneyness",
    "percentToBreakEvenBid",
    "delta",
    "impliedVolatilityRank1y",
    "potentialReturnAnnual",
    "potentialReturn",
    "underlyingLastPrice",
    "strike",
    "openInterest",
    "volume",
    "daysToExpiration",
]

# Gamma Exposure (GEX) metrics merged from gex101/ directory by date + symbol
GEX_FEATS = [
    "gex_total",
    "gex_total_abs",
    "gex_pos",
    "gex_neg",
    "gex_center_abs_strike",
    "gex_flip_strike",
    "gex_gamma_at_ul",
    "gex_distance_to_flip",
    "gex_sign_at_ul",
    "gex_missing",
]

# Macro / derived features: VIX, price momentum, normalised returns
NEW_FEATS = [
    "VIX",
    "ret_2d_norm",
    "ret_5d_norm",
    "prev_close_minus_ul_pct",
    "log1p_DTE",
]

# GEX subset used by winner model (excludes redundant columns)
WINNER_GEX_FEATS = [
    "gex_neg",
    "gex_center_abs_strike",
    "gex_total_abs",
]

# Lean feature set used by tail classifier — derived from feature-importance
# analysis on the old project; focuses on return/volatility regime signals.
# Earnings-window features are appended at runtime when WITH_EARNINGS=True.
TAIL_FEATS = [
    "potentialReturnAnnual",
    "VIX",
    "impliedVolatilityRank1y",
    "daysToExpiration",
    "underlyingLastPrice",
    "gex_center_abs_strike",
    "log1p_DTE",
    "prev_close_minus_ul_pct",
    "gex_neg",
    "potentialReturn",
    "gex_pos",
    "strike",
    "percentToBreakEvenBid",
    "gex_total_abs",
    "gex_flip_strike",
    "gex_gamma_at_ul",
]

# Optional earnings context features (appended to TAIL_FEATS when available)
TAIL_EARNINGS_FEATS = [
    "is_earnings_week",
    "is_earnings_window",
    "post_earnings_within_3d",
]

# 4-bin winner model probability features injected into the tail classifier.
# These are the per-class probabilities from the 4-bin winner OOF output, plus
# a derived conflict_score = p_bin0 * p_bin3 (high when the winner model is
# simultaneously uncertain between best and worst quartile — a strong tail signal).
BIN_PROB_FEATS = [
    "p_bin0",
    "p_bin1",
    "p_bin2",
    "p_bin3",
    "conflict_score",
]

# All features combined (training default when WINNER_FEATURES is not set)
ALL_FEATS = BASE_FEATS + GEX_FEATS + NEW_FEATS

# New GEX indicator features from gex101_indicator folder (see gex_ml_engine.md).
# All are numeric. "regime" is float64 (1.0 = positive, -1.0 = negative).
# "has_flip" is bool → cast to float (1.0/0.0) by fill_features.
# Note: "no_flip_flag" does NOT exist in the actual parquet files.
NEW_GEX_IND_FEATS = [
    "gamma_flip",
    "distance_to_flip",
    "flip_score",
    "gamma_density_near",
    "gamma_density_below",
    "support_asymmetry",
    "downside_void",
    "peak_concentration",
    "slope_near_price",
    "negative_gamma_below_ratio",
    "has_flip",
    "regime",   # float64: 1.0 = positive gamma regime, -1.0 = negative
]

# ---------------------------------------------------------------------------
# Calendar constants  (replaces magic numbers like `if weekday == 5`)
# ---------------------------------------------------------------------------
WEEKDAY_MONDAY    = 0
WEEKDAY_TUESDAY   = 1
WEEKDAY_WEDNESDAY = 2
WEEKDAY_THURSDAY  = 3
WEEKDAY_FRIDAY    = 4
WEEKDAY_SATURDAY  = 5
WEEKDAY_SUNDAY    = 6

# ---------------------------------------------------------------------------
# Data pipeline defaults
# ---------------------------------------------------------------------------
DEFAULT_TARGET_TIME         = "11:00"
DEFAULT_GLOB_PATTERN        = "coveredPut_*.csv"
DEFAULT_BATCH_SIZE          = 30          # symbols per yfinance download batch
DEFAULT_SLEEP_SECONDS       = 1.0         # pause between API calls
DEFAULT_GEX_WAIT_MINUTES    = 60          # re-fetch GEX only if older than this
MAX_DAYS_TO_EXPIRATION      = 14          # filter trades beyond this DTE
MIN_RATIO_CHANGE            = 0.1         # split-detection minimum price ratio change

# ---------------------------------------------------------------------------
# Labeling / scoring defaults
# ---------------------------------------------------------------------------
DEFAULT_TRAIN_EPSILON       = 0.0         # return_mon > epsilon => winner
DEFAULT_TAIL_FRACTION       = 0.05        # worst 5% by dollar PnL = tail loss
DEFAULT_WEIGHT_ALPHA        = 0.08
DEFAULT_WEIGHT_MIN          = 0.5
DEFAULT_WEIGHT_MAX          = 10.0

# ---------------------------------------------------------------------------
# Common start date for price data downloads
# ---------------------------------------------------------------------------
COMMON_START_DATE = "2025-04-01"
