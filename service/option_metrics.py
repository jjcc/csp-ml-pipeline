# options_metrics.py
import numpy as np
import pandas as pd

def _safe_div(a, b):
    """Elementwise safe division."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    mask = b != 0
    out[mask] = a[mask] / b[mask]
    return out

def prob_above(l, b, t_days, iv_annual):
    """
    Probability[S_T >= b] under lognormal w/ zero drift (per your formula):
      1 - Phi( ln(b/l) / (iv * sqrt(t/365)) )
    All inputs can be scalars or numpy arrays (broadcastable).
    """
    l = np.asarray(l, dtype=float)
    b = np.asarray(b, dtype=float)
    t_days = np.asarray(t_days, dtype=float)
    iv = np.asarray(iv_annual, dtype=float)

    # Handle degenerate cases
    iv = np.where(iv <= 0, np.nan, iv)
    t_years = t_days / 365.0
    t_years = np.where(t_years <= 0, np.nan, t_years)

    denom = iv * np.sqrt(t_years)
    z = np.log(_safe_div(b, l)) / denom
    # Standard normal CDF via error function
    Phi = 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))
    return 1.0 - Phi

def prob_below(l, b, t_days, iv_annual):
    """Probability[S_T < b] = 1 - prob_above."""
    return 1.0 - prob_above(l, b, t_days, iv_annual)

def compute_option_metrics(
    df: pd.DataFrame,
    price_col: str = "underlyingLastPrice",
    strike_col: str = "strike",
    bid_col: str = "bidPrice",
    dte_col: str = "daysToExpiration",
    iv_col: str = "impliedVolatility",  # annualized IV as a decimal, e.g. 0.55
    add_probabilities: bool = True,
    moneyness_denominator: str = "underlying",  # "underlying" (dataset) or "strike"
) -> pd.DataFrame:
    """
    Adds/overwrites the following columns (where possible):
      - moneyness
      - breakEvenBid
      - percentToBreakEvenBid
      - potentialReturn
      - potentialReturnAnnual
      - (if add_probabilities and iv_col present & valid)
          prob_above_strike
          prob_below_strike
          prob_above_breakeven
          prob_below_breakeven
          breakEvenProbability   (alias of prob_above_strike for naked puts)

    Notes:
      * percentToBreakEvenBid uses the observed convention in your examples:
        (breakEvenBid - price) / price * 100
      * potentialReturn is in % terms (not decimal).
      * IV must be annualized (e.g., 0.55 for 55%).
    """
    df = df.copy()

    # Pull arrays
    S = df[price_col].astype(float).to_numpy()
    K = df[strike_col].astype(float).to_numpy()
    bid = df[bid_col].astype(float).to_numpy()
    dte = df[dte_col].astype(float).to_numpy()

    # Derived basics
    # Moneyness (put convention): (K - S)/K ; positive = ITM, negative = OTM
    # --- FIX: moneyness denominator ---
    if moneyness_denominator == "underlying":
        denom = S
    elif moneyness_denominator == "strike":
        denom = K
    else:
        raise ValueError("moneyness_denominator must be 'underlying' or 'strike'")

    df["moneyness"] = np.round(_safe_div(K - S, denom),8)  # positive (ITM put) when K > S
    # Break-even (using bid)
    break_even = K - bid
    df["breakEvenBid"] = break_even

    # Percent to break-even (matches your sample): (BE - S)/S * 100
    df["percentToBreakEvenBid"] = np.round(_safe_div(break_even - S, S) * 100.0, 2)

    # Potential return % = bid / (K - bid) * 100  (guard K == bid)
    potentialReturn_br = _safe_div(bid, (K - bid)) * 100.0

    # Annualized potential return % = potentialReturn / DTE * 365
    df["potentialReturnAnnual"] = np.round(_safe_div(potentialReturn_br, dte) * 365, 3)

    # Potential return % = bid / (K - bid) * 100  (guard K == bid)
    df["potentialReturn"] = np.round(_safe_div(bid, (K - bid)) * 100.0, 5)

    # Probabilities (requires IV)
    if add_probabilities and iv_col in df.columns:
        iv = df[iv_col].astype(float).to_numpy()

        # Probability vs strike (OTM probability for puts: S_T >= K)
        p_above_strike = prob_above(S, K, dte, iv)
        p_below_strike = 1.0 - p_above_strike

        # Probability vs break-even (S_T >= (K - bid))
        p_above_be = prob_above(S, break_even, dte, iv)
        p_below_be = 1.0 - p_above_be

        df["prob_above_strike"] = p_above_strike
        df["prob_below_strike"] = p_below_strike
        df["prob_above_breakeven"] = p_above_be
        df["prob_below_breakeven"] = p_below_be

        # Many feeds label "OTM Probability" or "BreakEvenProbability" for puts as P(S_T >= K)
        df["breakEvenProbability"] = p_above_strike

    return df
