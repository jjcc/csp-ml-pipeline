from pathlib import Path
import pandas as pd
import numpy as np
import glob
import math
import re
import os


FILENAME_RE = re.compile(r'^(?P<symbol>[A-Za-z0-9.\-_]+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<hour>\d{2})-(?P<minute>\d{2})\.csv$')

def parse_timestamp_from_filename(name: str):
    """
    Expecting coveredPut_YYYY-MM-DD_HH_MM.csv or similar.
    Returns (date, time) as datetime.date, datetime.time or (None,None) if not parsed.
    """
    import re, datetime as dt
    m = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})', name)
    if not m:
        return None, None
    d = dt.datetime.strptime(m.group(1), "%Y-%m-%d").date()
    t = dt.time(int(m.group(2)), int(m.group(3)))
    return d, t

def pick_daily_snapshot_files(data_dir: str, pattern: str, target_time_str: str = "11:00"):
    """
    Groups files by date and picks one file per date:
    - choose the file with time >= target_time that is closest to it
    - if none, fall back to the latest file before target_time
    Returns a sorted list of absolute paths.
    """
    import glob, os, datetime as dt
    target_h, target_m = map(int, target_time_str.split(":"))
    target = dt.time(target_h, target_m)

    #paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    paths = sorted([str(Path(p)) for p in Path(data_dir).glob(pattern)])
    by_date = {}
    for p in paths:
        name = os.path.basename(p)
        d, t = parse_timestamp_from_filename(name)
        if d is None or t is None:
            continue
        by_date.setdefault(d, []).append((t, p))

    chosen = []
    for d, lst in by_date.items():
        lst.sort()  # sort by time
        after = [x for x in lst if x[0] >= target]
        if after:
            # pick nearest after target
            chosen.append(min(after, key=lambda x: (dt.datetime.combine(d, x[0]) - dt.datetime.combine(d, target)).total_seconds())[1])
        else:
            # pick latest before target
            chosen.append(max(lst, key=lambda x: x[0])[1])
    chosen = [str(Path(p)) for p in chosen]
    return sorted(chosen)

def filter_by_dte(df, dte_weekday_map = {4:0, 3:1}):
    """
    Docstring for filter_by_dte

    :param df: input dataframe
    :param dte_weekday_map: mapping of days to expiration (DTE) to expected capture weekday
        {4:0, 3:1} means DTE 4 should be captured on Monday (0), DTE 3 on Tuesday (1)
        {11:0, 10:1} means DTE 11 should be captured on Monday (0), DTE 10 on Tuesday (1)
    :return: filtered dataframe
    """

    dtes = [dte for dte in dte_weekday_map]
    weekdays = [dte_weekday_map[dte] for dte in dtes]
    df_filtered = df[df["daysToExpiration"].isin(dtes)]
    df_filtered = df_filtered[df_filtered["captureWeekday"].isin(weekdays)]
    return df_filtered

def keep_one_row_per_contract_per_day(
            df,
            contract_id_col="symbol",
            dt_col="captureTime",
            target_time="11:00",
        ):
    df = df.copy()

    tgt = int(target_time.split(":")[0]) * 60 + int(target_time.split(":")[1])

    df["_date"] = df[dt_col].dt.date
    df["_tod_min"] = df[dt_col].dt.hour * 60 + df[dt_col].dt.minute
    df["_time_diff"] = (df["_tod_min"] - tgt).abs()

    df = (
        df.sort_values([contract_id_col, "_date", "_time_diff", dt_col], kind="mergesort")
          .groupby([contract_id_col, "_date"], as_index=False)
          .first()
    )

    return df.drop(columns=["_tod_min", "_time_diff", "_date"], errors="ignore")


def load_csp_files(data_dir: str, pattern: str, target_time="11:00", enforce_daily_pick=True, cut_off_date=None) -> pd.DataFrame:
    if enforce_daily_pick:
        paths = pick_daily_snapshot_files(data_dir, pattern, target_time)
    else:
        import glob, os
        paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    frames = []
    for idx, p in enumerate(paths):
        try:
            df = pd.read_csv(p)
            df = df.drop(columns=["baseSymbolType","Unnamed: 0", "symbolType"], errors='ignore')
            base_file = Path(p).name
            df["__source_file"] = base_file
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}")
        if (idx + 1) % 100 == 0:
            print(f"[INFO] --load_csp_files(): Loaded {idx+1}/{len(paths)} files")

    if not frames:
        raise SystemExit(f"No files found for pattern {pattern} in {data_dir}")
    df_all = pd.concat(frames, ignore_index=True)
    # additional processing to add captureTime column so that we can use later
    df_all['capture_time'] = df_all['__source_file'].apply(
        lambda x: (
            [x.split('.')[0].split('_')[-3], ':'.join(x.split('.')[0].split('_')[-2:])]
            if len(x.split('.')[0].split('_')) >= 3
            else [None, None]
        )
    )
    df_all['captureTime'] = pd.to_datetime(df_all['capture_time'].apply(lambda x: f"{x[0]} {x[1]}:00"), errors='coerce')
    df_all['captureWeekday'] = pd.to_datetime(df_all['capture_time'].apply(lambda x: x[0]), errors='coerce').dt.weekday
    # Need to keep only one row per contract per day
    df_all = keep_one_row_per_contract_per_day(
        df_all,
        contract_id_col="symbol",
        dt_col="captureTime",
        target_time=target_time,
    )
    return df_all


def compute_gex_features(df_gex: pd.DataFrame, ul_price: float) -> dict:
    out = {
        "gex_total": np.nan,
        "gex_total_abs": np.nan,
        "gex_pos": np.nan,
        "gex_neg": np.nan,
        "gex_center_abs_strike": np.nan,
        "gex_flip_strike": np.nan,
        "gex_gamma_at_ul": np.nan,
        "gex_distance_to_flip": np.nan,
        "gex_sign_at_ul": np.nan,
    }
    if df_gex is None or df_gex.empty:
        return out

    g = df_gex.copy()
    if "strike" not in g.columns or "gamma" not in g.columns:
        return out
    g = g[["strike", "gamma"]].dropna()
    if g.empty:
        return out
    g["strike"] = pd.to_numeric(g["strike"], errors="coerce")
    g["gamma"] = pd.to_numeric(g["gamma"], errors="coerce")
    g = g.dropna()
    if g.empty:
        return out
    g = g.sort_values("strike")

    total = g["gamma"].sum()
    total_abs = g["gamma"].abs().sum()
    pos = g.loc[g["gamma"] > 0, "gamma"].sum()
    neg = g.loc[g["gamma"] < 0, "gamma"].sum()

    out["gex_total"] = float(total)
    out["gex_total_abs"] = float(total_abs)
    out["gex_pos"] = float(pos)
    out["gex_neg"] = float(neg)

    if total_abs > 0:
        out["gex_center_abs_strike"] = float((g["strike"] * g["gamma"].abs()).sum() / total_abs)

    cum = g["gamma"].cumsum().values
    strikes = g["strike"].values
    flip = np.nan
    for i in range(1, len(cum)):
        if np.sign(cum[i-1]) != np.sign(cum[i]):
            x1, y1 = strikes[i-1], cum[i-1]
            x2, y2 = strikes[i], cum[i]
            if (y2 - y1) != 0:
                t = -y1 / (y2 - y1)
                flip = x1 + t * (x2 - x1)
            else:
                flip = strikes[i]
            break
    out["gex_flip_strike"] = float(flip) if not (isinstance(flip, float) and math.isnan(flip)) else np.nan

    if ul_price is not None and not np.isnan(ul_price):
        try:
            out["gex_gamma_at_ul"] = float(np.interp(ul_price, strikes, g["gamma"].values))
            out["gex_sign_at_ul"] = float(np.sign(out["gex_gamma_at_ul"]))
        except Exception:
            pass

    if pd.notna(out["gex_flip_strike"]) and ul_price is not None and not np.isnan(ul_price):
        out["gex_distance_to_flip"] = float(out["gex_flip_strike"] - ul_price)

    # for all the columns, round to 6 decimal places
    for k in out.keys():
        if isinstance(out[k], float) and not math.isnan(out[k]):
            out[k] = round(out[k], 6)

    return out

def merge_gex(trades, base_dir, target_minutes):
    need = ["baseSymbol","tradeTime","underlyingLastPrice"]
    for c in need:
        if c not in trades.columns:
            raise SystemExit(f"Missing required column: {c}")
    trades["tradeTime"] = pd.to_datetime(trades["tradeTime"], errors="coerce")
    trades["trade_date"] = trades["tradeTime"].dt.strftime("%Y-%m-%d")
    trades["symbol_norm"] = trades["baseSymbol"].astype(str).str.lower()

    cache = {}
    feats = []

    for idx, row in trades.iterrows():
        sym = row["symbol_norm"]
        d = row["trade_date"]
        ul = float(row["underlyingLastPrice"]) if pd.notna(row["underlyingLastPrice"]) else np.nan
        key = (sym, d)
        if key not in cache:
            day_dir = Path(base_dir) / d
            if not day_dir.exists():
                cache[key] = (None, None)
            else:
                pattern = str(day_dir / f"{sym}_{d}_*.csv")
                cand = glob.glob(pattern)
                if cand:
                    chosen = pick_closest_file(cand, target_minutes)
                    try:
                        gex_df = pd.read_csv(chosen)
                    except Exception:
                        gex_df = None
                    cache[key] = (gex_df, chosen)
                else:
                    cache[key] = (None, None)

        gex_df, chosen_path = cache[key]
        feat = compute_gex_features(gex_df, ul)
        feat["gex_file"] = chosen_path if chosen_path else ""
        feat["gex_missing"] = 0 if gex_df is not None and not gex_df.empty else 1
        feats.append(feat)
        if (idx + 1) % 100 == 0:
            print(f"[INFO] Processed {idx+1}/{len(trades)} rows")

    gex_df_all = pd.DataFrame(feats, index=trades.index)
    merged = pd.concat([trades, gex_df_all], axis=1)
    return merged

def parse_gex_filename(name: str):
    m = FILENAME_RE.match(name)
    if not m:
        return None
    dd = m.group('date')
    hh = int(m.group('hour')); mm = int(m.group('minute'))
    return {
        "symbol": m.group('symbol').lower(),
        "date": dd,
        "stamp": f"{hh:02d}:{mm:02d}",
        "minutes": hh * 60 + mm,
    }



def pick_closest_file(files, target_minutes: int):
    best = None; best_dist = 10**9
    for p in files:
        meta = parse_gex_filename(os.path.basename(p))
        if not meta:
            continue
        dist = abs(meta["minutes"] - target_minutes)
        if dist < best_dist:
            best = p; best_dist = dist
    return best


def add_dte_and_normalized_returns(df, non_lableled=False):
    """
    Add DTE and normalized returns to the dataframe.
    Originally from train_tail_with_gex.py
    """
    d = df.copy()
    for c in ("tradeTime","expirationDate"):
        if c in d.columns:
            d[c] = pd.to_datetime(d[c], errors="coerce")

    # The field daysToExpiration is supposed to exist
    #d["daysToExpiration"] = ((d["expirationDate"].dt.floor("D") - d["tradeTime"].dt.floor("D"))
    #                          .dt.days.clip(lower=1))
    #d["log1p_DTE"] = np.log1p(d["daysToExpiration"].astype(float))
    if non_lableled: # no callculation to returns
        return d

    d["return_per_day"] = d["return_pct"] / d["daysToExpiration"].replace(0, 1)
    #d["return_ann"] = ((1.0 + d["return_pct"] / 100.0) ** (365.0 / d["daysToExpiration"]) - 1.0) * 100.0
    d["return_ann"] =d["return_pct"] * 365.0 / d["daysToExpiration"].replace(0, 1)
    d["return_mon"] =d["return_pct"] * 30.0 / d["daysToExpiration"].replace(0, 1)
    return d
