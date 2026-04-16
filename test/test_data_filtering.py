import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service.preprocess import filter_by_dte
'''
There is a big issue that the raw data contains multiple entries for the same contract on different trading time.
It's important to keep the contract uniqiue.

The strategies to handle this issue include:
1. Combine DTE with first trade time to make contract unique
2. Combine DTE with close to certain time (e.g., 11:00) to make contract unique
3. Keep only the first trade entry for each contract'''

OUTPUT_FOLDER = "output"
DATA_FOLDER = f"{OUTPUT_FOLDER}/data_prep"


from datetime import time as dtime

def _hhmm_to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)

def select_one_row_per_contract_dte(
    df,
    allowed_dtes=(3, 4),
    contract_id_col="symbol",
    dte_col="daysToExpiration",
    dt_col="captureTime",
    weekday_col="captureWeekday",
    expected_weekday_map=None,     # e.g. {4:"Mon", 3:"Tue"}  (optional)
    target_time_hhmm="11:00",      # pick row closest to this time
    tie_break="earliest",          # or "latest"
):
    df = df.copy()

    # 1) Restrict to allowed DTEs
    #df = df[df[dte_col].isin(list(allowed_dtes))]

    ## 2) Optional guardrail: require expected weekday for each DTE
    #if expected_weekday_map:
    #    ok = False
    #    # Build boolean mask for allowed (dte, weekday)
    #    mask = None
    #    for dte, wd in expected_weekday_map.items():
    #        m = (df[dte_col].eq(dte) & df[weekday_col].eq(wd))
    #        mask = m if mask is None else (mask | m)
    #    df = df[mask] if mask is not None else df.iloc[0:0]

    #if df.empty:
    #    return df

    # 3) Collapse intraday duplicates per (contract, dte) deterministically
    tgt = _hhmm_to_minutes(target_time_hhmm)
    # assume dt_col is datetime-like
    df["_tod_min"] = df[dt_col].dt.hour * 60 + df[dt_col].dt.minute
    df["_time_diff"] = (df["_tod_min"] - tgt).abs()

    # deterministic ordering inside each group
    ascending_dt = True if tie_break == "earliest" else False
    df = (
        df.sort_values([contract_id_col, dte_col, "_time_diff", dt_col],
                       ascending=[True, True, True, ascending_dt],
                       kind="mergesort")
          .groupby([contract_id_col, dte_col], as_index=False)
          .first()
    )

    return df.drop(columns=["_tod_min", "_time_diff"], errors="ignore")



class TestDataFiltering(unittest.TestCase):

    def test_check_dataa(self):
        '''
        '''
        file1 = "trades_raw_orig.csv"
        #file1 = "trades_raw_b_0901.csv"
        file1 = f"{DATA_FOLDER}/{file1}"
        df = pd.read_csv(file1)
        # source file column looks like: "coveredPut_2025-06-21_11_00.csv"
        # create a new column 'capture_time' by extracting date and time from '_source_file'
        df['_source_file'] = df['__source_file'].astype(str)
        df['capture_time'] = df['_source_file'].apply(
            lambda x: (
                [x.split('.')[0].split('_')[-3], ':'.join(x.split('.')[0].split('_')[-2:])]
                if len(x.split('.')[0].split('_')) >= 3
                else [None, None]
            )
        )
        df['captureWeekday'] = pd.to_datetime(df['capture_time'].apply(lambda x: x[0]), errors='coerce').dt.weekday
        df['captureTime'] = pd.to_datetime(df['capture_time'].apply(lambda x: f"{x[0]} {x[1]}:00"), errors='coerce')


        df_fitlred = filter_by_dte(df)
        print(f"Original df shape: {df.shape}, Filtered df shape: {df_fitlred.shape}")
        df_unique = select_one_row_per_contract_dte(
            df_fitlred,
            allowed_dtes=[3,4],
            contract_id_col="symbol",
            dte_col="daysToExpiration",
            dt_col="captureTime",
            weekday_col="captureWeekday",
            expected_weekday_map={4:0, 3:1},   # DTE 4 on Mon (0), DTE 3 on Tue (1)
            target_time_hhmm="11:00",
            tie_break="earliest",
        )
        print(f"Unique df shape: {df_unique.shape}")




        #df['trading_time'] = pd.to_datetime(df['trading_time'].apply(lambda x: f"{x[0]} {x[1]}:{x[2]}:00"), errors='coerce')
        pass




    def test_get_config(self):
        from service.env_config import getenv
        data_dir = getenv("COMMON_DATA_DIR")
        basic_csv = getenv("COMMON_DATA_BASIC_CSV", "labeled_trades_normal.csv")
        output_dir = getenv("COMMON_OUTPUT_DIR", "./output")
        output_csv = getenv("COMMON_OUTPUT_CSV", "labeled_trades.csv")
        self.assertIsNotNone(data_dir)
        self.assertIsNotNone(basic_csv)
        self.assertIsNotNone(output_dir)
        self.assertIsNotNone(output_csv)
        print(f"Data dir: {data_dir}, Basic CSV: {basic_csv}")
        print(f"Output dir: {output_dir}, Output CSV: {output_csv}")


    def test_filter_earning_proximity(self):
        from service.nasdaq_earnings import add_earnings_proximity
        #file1 = "labeled_trades_with_gex_macro_f_1027.csv"
        file1 = "labeled_trades_with_gex_macro_orig.csv"
        file1 = f"{OUTPUT_FOLDER}/data_labeled/{file1}"
        df_opt = pd.read_csv(file1)
        df_opt = df_opt[df_opt['won'] == False]

        df_earning = pd.read_csv("data/earnings_surprise_orig.csv")

        df_opt = add_earnings_proximity(df_opt, df_earning)

        df_close_to_earnings = df_opt[df_opt['days_to_nearest_earnings'].abs() <= 7]
        print(f"Filtered df shape (close to earnings within 7 days): {df_close_to_earnings.shape}")

        print(f"Original df shape: {df_opt.shape}")
        symbols = df_close_to_earnings['baseSymbol'].unique()
        print(f"Symbols close to earnings: {symbols}")
        
    
    def test_get_lose_symbols(self):
        file1 = "labeled_trades_with_gex_macro_orig.csv"
        file1 = f"{OUTPUT_FOLDER}/data_labeled/{file1}"
        df = pd.read_csv(file1)
        df = df[df['won'] == False]
        lose_symboles = df['baseSymbol'].unique()
        print(f"Lose symbols: {lose_symboles}")



if __name__ == '__main__':
    unittest.main()
