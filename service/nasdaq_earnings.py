from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

try:
    from diskcache import Cache
except ImportError:
    Cache = None  # optional


# -----------------------------
# Schema (Pydantic)
# -----------------------------

class EarningsSurpriseRow(BaseModel):
    """
    Nasdaq returns a 'rows' array where each row is a dict (keys may vary a bit).
    We validate the presence of a minimal, stable subset, and allow extra fields.
    """
    model_config = ConfigDict(extra="allow")

    # Typical keys we expect to exist (names sometimes vary)
    fiscal_quarter_ending: Optional[str] = Field(default=None, alias="fiscalQuarterEnding")
    date_reported: Optional[str] = Field(default=None, alias="dateReported")
    eps_forecast: Optional[str] = Field(default=None, alias="epsForecast")
    eps_actual: Optional[str] = Field(default=None, alias="epsActual")
    surprise_percent: Optional[str] = Field(default=None, alias="surprisePercent")


class EarningsSurpriseData(BaseModel):
    model_config = ConfigDict(extra="allow")

    rows: List[EarningsSurpriseRow] = Field(default_factory=list)


class NasdaqEarningsSurpriseResponse(BaseModel):
    """
    We validate the core structure:
      { data: { rows: [...] }, status: {...} }
    If Nasdaq changes shape, you'll fail fast with a clear error.
    """
    model_config = ConfigDict(extra="allow")

    data: Optional[EarningsSurpriseData] = None
    message: Optional[str] = None


# -----------------------------
# Exceptions
# -----------------------------

class NasdaqScrapeError(RuntimeError):
    pass


class NasdaqBlockedError(NasdaqScrapeError):
    pass


class NasdaqBadResponseError(NasdaqScrapeError):
    pass


# -----------------------------
# Client
# -----------------------------

@dataclass
class NasdaqClient:
    timeout_s: float = 15.0
    cache_dir: Optional[str] = None  # e.g. ".cache_nasdaq"
    cache_ttl_s: int = 24 * 3600     # 1 day

    def __post_init__(self) -> None:
        self.session = requests.Session()

        # "Browser-ish" headers reduce blocking
        self.base_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nasdaq.com/",
            "Origin": "https://www.nasdaq.com",
            "Connection": "keep-alive",
        }

        self.cache = None
        if self.cache_dir and Cache is not None:
            self.cache = Cache(self.cache_dir)

    def close(self) -> None:
        if self.cache is not None:
            self.cache.close()
        self.session.close()

    def _cache_get(self, key: str) -> Optional[dict]:
        if self.cache is None:
            return None
        return self.cache.get(key)

    def _cache_set(self, key: str, value: dict) -> None:
        if self.cache is None:
            return
        self.cache.set(key, value, expire=self.cache_ttl_s)

    @retry(
        reraise=True,
        stop=stop_after_attempt(6),
        wait=wait_exponential_jitter(initial=1, max=20),
        retry=(
            retry_if_exception_type(requests.RequestException)
            | retry_if_exception_type(NasdaqBlockedError)
            | retry_if_exception_type(NasdaqBadResponseError)
        ),
    )
    def _get_json(self, url: str) -> Dict[str, Any]:
        resp = self.session.get(url, headers=self.base_headers, timeout=self.timeout_s)

        # Common blocking patterns: 403/429, or HTML instead of JSON
        if resp.status_code in (403, 429):
            raise NasdaqBlockedError(f"Blocked by Nasdaq (HTTP {resp.status_code})")

        ct = (resp.headers.get("Content-Type") or "").lower()
        if "application/json" not in ct:
            # Sometimes they return HTML challenge pages
            raise NasdaqBadResponseError(f"Expected JSON but got Content-Type={ct}")

        try:
            return resp.json()
        except json.JSONDecodeError as e:
            raise NasdaqBadResponseError(f"JSON decode failed: {e}") from e

    def earnings_surprise_history(self, symbol: str, force_refresh: bool = False) -> List[dict]:
        """
        Returns list[dict] rows from Nasdaq earnings-surprise endpoint.
        Each row includes (when available):
          fiscalQuarterEnding, dateReported, epsForecast, epsActual, surprisePercent, ...
        """
        symbol = symbol.strip().upper()
        cache_key = f"nasdaq:earnings-surprise:{symbol}"

        if not force_refresh:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached["rows"]

        url = f"https://api.nasdaq.com/api/company/{symbol}/earnings-surprise"

        raw = self._get_json(url)

        # Schema validation
        try:
            parsed = NasdaqEarningsSurpriseResponse.model_validate(raw)
        except ValidationError as e:
            raise NasdaqBadResponseError(f"Schema validation failed for {symbol}: {e}") from e

        rows: List[dict] = []
        if parsed.data and (parsed.data.rows or parsed.data.model_extra['earningsSurpriseTable']):
            # Convert back to plain dicts with original keys preserved where possible
            if parsed.data.model_extra.get('earningsSurpriseTable'):
                for r in parsed.data.model_extra['earningsSurpriseTable']["rows"]:
                    rows.append(r)
            if parsed.data.rows:
                for r in parsed.data.rows:
                    rows.append(r.model_dump(by_alias=True))
        else:
            # Nasdaq sometimes returns message fields for invalid symbols or no data
            msg = parsed.message or "No data rows returned"
            raise NasdaqBadResponseError(f"{symbol}: {msg}")

        self._cache_set(cache_key, {"rows": rows, "fetched_at": int(time.time())})
        return rows

import pandas as pd

def earnings_rows_to_df(symbol: str, rows: list[dict]) -> pd.DataFrame:
    """
    rows example:
    {'fiscalQtrEnd': 'Sep 2025', 'dateReported': '11/4/2025', ...}
    """
    df = pd.DataFrame(rows).copy()
    df["symbol"] = symbol.upper()

    # Parse US-style m/d/YYYY
    df["earnings_date"] = pd.to_datetime(df["dateReported"], format="%m/%d/%Y", errors="coerce")

    # Keep only valid parsed dates
    df = df.dropna(subset=["earnings_date"])

    # (Optional) drop duplicates, keep most recent if duplicates
    df = df.drop_duplicates(subset=["symbol", "earnings_date"]).sort_values(["symbol", "earnings_date"])
    return df[["symbol", "earnings_date"]]

def add_earnings_proximity(
    opt_df: pd.DataFrame,
    earn_df: pd.DataFrame,
    sym_col="baseSymbol",
    expiry_col="expirationDate"
) -> pd.DataFrame:
    """
    For each option in opt_df, find the nearest earnings date before and after expiry,
    and compute days to nearest earnings.
    Args:
        opt_df: DataFrame with options data, must have columns for symbol and expiry date
        earn_df: DataFrame with earnings dates, must have columns 'symbol' and 'earnings_date'
        sym_col: Column name in opt_df for the underlying symbol
        expiry_col: Column name in opt_df for the option expiry date
    Returns:
        DataFrame with added columns:
          'earn_prev' - nearest earnings date before expiry
          'earn_next' - nearest earnings date after expiry
          'days_to_nearest_earnings' - days to nearest earnings date
    """
    out = opt_df.copy()
    out[sym_col] = out[sym_col].astype(str).str.upper()

    out["expiry_dt"] = pd.to_datetime(out[expiry_col], errors="coerce")
    out = out.dropna(subset=["expiry_dt", sym_col])

    e = earn_df.copy()
    e.rename(columns={"symbol": sym_col}, inplace=True)
    e["earnings_date"] = pd.to_datetime(e["earnings_date"], errors="coerce")
    # merge_asof requires both sides sorted by the merge key and group
    e = e.sort_values([sym_col, "earnings_date"], kind="mergesort")

    orig_index = out.index
    # Prepare for merge_asof: both sides must be sorted by group then merge key
    out = out.sort_values([sym_col, "expiry_dt"], kind="mergesort")

    # Nearest earnings BEFORE expiry
    prev_e_right = (
        e.rename(columns={"symbol": sym_col, "earnings_date": "earn_prev"})
        .sort_values([sym_col, "earn_prev"], kind="mergesort")
    )
    prev_e = pd.merge_asof(
        out,
        prev_e_right,
        by=sym_col,
        left_on="expiry_dt",
        right_on="earn_prev",
        direction="backward",
        allow_exact_matches=True,
    )

    # Nearest earnings AFTER expiry
    next_e_right = (
        e.rename(columns={"symbol": sym_col, "earnings_date": "earn_next"})
        .sort_values([sym_col, "earn_next"], kind="mergesort")
    )
    next_e = pd.merge_asof(
        out,
        next_e_right,
        by=sym_col,
        left_on="expiry_dt",
        right_on="earn_next",
        direction="forward",
        allow_exact_matches=True,
    )

    # Compute distances (days)
    d_prev = (prev_e["expiry_dt"] - prev_e["earn_prev"]).dt.days.abs()
    d_next = (next_e["earn_next"] - next_e["expiry_dt"]).dt.days.abs()

    out["earn_prev"] = prev_e["earn_prev"]
    out["earn_next"] = next_e["earn_next"]
    out["days_to_nearest_earnings"] = pd.concat([d_prev, d_next], axis=1).min(axis=1)

    return out.loc[orig_index]


# -----------------------------
# Quick CLI / demo
# -----------------------------
#if __name__ == "__main__":
#    list_loss = "BA,BULL,BYND,CCJ,CCL,CELH,CHWY,CIFR,CLF,CLOV,CMG,COIN,CORZ,COST,CRM,DASH,DELL,DJT,DKNG,HIVE,HOOD,IREN,JBLU,JOBY,KHC,KULR,MDLZ,META,MOS,OKLO,ONDS,ON,OPEN,ORCL,OSCR,OXY,PANW,PATH,PCT,PINS,PLTR,PM,RGTI,RILY,RIOT,RKLB,RR,RUM,RXRX,SBET,T,UAMY,WFC,WULF,XYZ,ZETA"
#    list_loss = list_loss.split(",")
#    c = NasdaqClient(cache_dir="data/.cache_nasdaq")
#    #for symbol in ["NVTS", "CMG", "META","SMR","CLOV","CRWV","UEC","UUUU","TEM"]:
#    earning_by_symbol = {}
#    for symbol in list_loss:
#        try:
#            rows = c.earnings_surprise_history(symbol)
#            print(f"Rows for {symbol}: {len(rows)}")
#            print(rows[0] if rows else "No rows")
#        except NasdaqScrapeError as e:
#            print(f"Error for {symbol}: {e}")
#        finally:
#            c.close()
#
#        df = earnings_rows_to_df(symbol, rows)
#        time.sleep(1)  # be nice to Nasdaq servers
#        earning_by_symbol[symbol] = df
#
#    all_earnings_df = pd.concat(earning_by_symbol.values(), ignore_index=True)
#    all_earnings_df = all_earnings_df.sort_values(["symbol", "earnings_date"])
#    all_earnings_df.to_csv("data/earnings_surprise_f.csv", index=False)
