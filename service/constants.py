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

# GEX subset used by winner and tail models (excludes redundant columns)
WINNER_GEX_FEATS = [
    "gex_neg",
    "gex_center_abs_strike",
    "gex_total_abs",
]

# All features combined (training default when WINNER_FEATURES is not set)
ALL_FEATS = BASE_FEATS + GEX_FEATS + NEW_FEATS

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
