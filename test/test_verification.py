import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import re
import shutil
import unittest
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from service.data_prepare import derive_capital, lookup_close_on_or_before, preload_prices_with_cache



# Use preloaded prices to compute expiry_close
def expiry_close_from_cache(r, preload_closes):
    if preload_closes is None:
        return np.nan
    
    if 'expirationDate' not in r:
        symbol = str(r['symbol']).upper()
        _, expiry_date, _ = symbol.split('|')
        expiry_date = expiry_date[:4] + '-' + expiry_date[4:6] + '-' + expiry_date[6:]
        r['expirationDate'] = expiry_date


    # Don't label future expiration dates
    expiry_date = pd.to_datetime(r['expirationDate'])
    if pd.isna(expiry_date) or expiry_date > pd.Timestamp.now():
        return np.nan

    sym = str(r['baseSymbol']).upper()
    price_df = preload_closes.get(sym)
    # Only allow prices within 3 trading days of expiration
    return lookup_close_on_or_before(price_df, expiry_date, max_days_back=3) if price_df is not None else np.nan

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def entry_credit(r,  min_abs=0.01):
    bidPrice = safe_float(r["bidPrice"])
    fill = bidPrice - min_abs
    return fill*100.0

class TestVerification(unittest.TestCase):

    def setUp(self):
        pass

    def test_verify_real_return_calculation(self):
        """
        """
        # 'Unnamed: 0', 'symbol', 'baseSymbol', 'baseSymbolType', 'underlyingLastPrice', 'expirationDate', 
        # 'daysToExpiration', 'strike', 'moneyness', 'bidPrice', 'breakEvenBid', 'percentToBreakEvenBid', 'volume', 'openInterest', 'impliedVolatilityRank1y', 
        # 'delta', 'potentialReturn', 'potentialReturnAnnual', 'breakEvenProbability', 'expirationType', 'symbolType', 'tradeTime',
        #  '__source_file', 'bid', 'ask', 'expiry_close', 'win', 'entry_credit', 'exit_intrinsic', 'capital', 'total_pnl', 'return_pct', 'trade_date', 'symbol_norm', 'gex_total', 'gex_total_abs', 'gex_pos', 'gex_neg', 'gex_center_abs_strike', 'gex_flip_strike', 'gex_gamma_at_ul', 'gex_distance_to_flip', 'gex_sign_at_ul', 'gex_file', 'gex_missing', 'log1p_DTE', 'VIX', 'prev_close', 'ret_2d', 'ret_5d', 'ret_2d_norm', 'ret_5d_norm', 'prev_close_minus_strike', 'prev_close_minus_strike_pct', 'tail_proba', 'is_tail_preda'
        file = "output/tails_score/scored_with_tail_pct_mon_cut5.csv"
        col_to_keep = ["symbol", "baseSymbol", "underlyingLastPrice", "expirationDate", "daysToExpiration", "bidPrice",
                       "strike", 'potentialReturn','potentialReturnAnnual','tradeTime','expiry_close','entry_credit','exit_intrinsic',
                       'capital', 'total_pnl','return_pct']
        df = pd.read_csv(file)
        #cols = list(df.columns)
        #print("Columns in the CSV file:", cols)
        df = df[col_to_keep]
        output_file = "output/tails_score/scored_with_tail_pct_mon_cut5_short.csv"
        df.to_csv(output_file, index=False)
        # Check if both functions produce the same result
        #pd.testing.assert_frame_equal(result1, result2)

        

        #
        #assert(medians_x == medians_t)

    def test_drop_bidask(self):
        file = "output/labeled_trades_normal_gex_macro2.csv"
        df = pd.read_csv(file)
        df = df.drop(columns=["bid", "ask"], errors="ignore")
        #output_file = "output/labeled_trades_with_gex_normal_dropped.csv"
        output_file = file
        df.to_csv(output_file, index=False)

    def test_copy_monday_results(self):
        """
        Copy only monday results to special folder. For a simple afterwards verification.
        """
        load_dotenv(".env.test")
        folder_check = "prod/output"
        files = glob.glob(os.path.join(folder_check, "*.csv"))
        print(f"Found {len(files)} files")
        monday_files = []
        for file in files:
            file_name = file.split("/")[-1]
            date_segment, weekday = self.get_date_seg(file_name)
            if weekday == "Monday":
                #target_folder = os.path.join(folder_check, "monday_only")
                #os.makedirs(target_folder, exist_ok=True)
                #target_file = os.path.join(target_folder, file_name)
                #print(f"Copying {file} to {target_file}")
                #shutil.copy(file, target_file)
                monday_files.append(file_name)
        
        for mf in monday_files:
            target_folder = os.path.join(folder_check, "monday_only")
            os.makedirs(target_folder, exist_ok=True)
            target_file = os.path.join(target_folder, mf)
            mf = os.path.join(folder_check, mf)
            shutil.copy(mf, target_file)
            print(f"Copying {mf} to {target_file}")


    def get_date_seg(self, file_name):
        date_segment = re.search(r'[\w|_]+(\d{4}-\d{2}-\d{2})_\d{2}_\d{2}\.csv', file_name)
        if date_segment:
            date_segment = date_segment.group(1)
        if not date_segment:
            return None, None
        # get the weekday of that date
        weekday = pd.to_datetime(date_segment).day_name()
        return date_segment, weekday


    def test_fill_expiry_close(self):
        """
        Fill expiry_close column for all files in monday_only folder.
        The result file will be saved in labeled folder.
        """
        load_dotenv(".env.test")
        monday_only_dir = os.getenv("MONDAY_ONLY_DIR", "prod/output/monday_only")
        files = glob.glob(os.path.join(monday_only_dir, "*.csv"))
        print(f"Found {len(files)} files to process expiry_close")
        for file in files:
            print(f"Processing expiry_close for {file}")
            self.fill_expiry_close(file)

    def test_calculate_pnl(self):
        """
        Calculate entry_credit, exit_intrinsic, total_pnl, return_pct columns for all files in labeled folder.
        The result file will overwrite the original file.
        """
        load_dotenv(".env.test")
        labeled_output_dir = os.getenv("LABELED_OUTPUT_DIR", "prod/output/labeled")



        files = glob.glob(os.path.join(labeled_output_dir, "*.csv"))
        print(f"Found {len(files)} files to process PnL calculation")

        for file in files:
            print(f"Processing PnL calculation for {file}")
            df = pd.read_csv(file)

            # Entry credit for puts
            df["entry_credit"] = df.apply(entry_credit, axis=1)

            # fill strike if not exist
            if 'strike' not in df.columns:
                df['strike'] = df['symbol'].apply(
                    #lambda s: str(s).upper()[str(s).upper().find('|')+10:str(s).upper().find('|')+15]
                    lambda s: str(s).split('|')[2][:-1]
                ).apply(lambda x: safe_float(x))


            # Exit (expiry intrinsic) for puts
            def exit_intrinsic(r):
                strike = safe_float(r["strike"]) 
                expiry_close = safe_float(r["expiry_close"])
                if not np.isfinite(strike) or not np.isfinite(expiry_close):
                    return np.nan
                return max(0.0, strike - expiry_close)*100.0

            df["exit_intrinsic"] = df.apply(exit_intrinsic, axis=1)

            # Capital reserved for CSP
            df["capital"] = derive_capital(df)

            # Total PnL and return
            df["total_pnl"] = df["entry_credit"] - df["exit_intrinsic"]
            df["return_pct"] = np.where(df["capital"]>0, df["total_pnl"]/df["capital"]*100.0, np.nan)
            df["win_verdict"] = np.where(df["total_pnl"]>0, 1, 0)

            df.to_csv(file, index=False)    


    def test_collect_stats(self):
        """
        Collect basic stats from all labeled files.
        """
        load_dotenv(".env.test")
        labeled_output_dir = os.getenv("LABELED_OUTPUT_DIR", "prod/output/labeled")

        files = glob.glob(os.path.join(labeled_output_dir, "score*.csv"))
        print(f"Found {len(files)} files to collect stats")

        stats = []
        for file in files:
            df = pd.read_csv(file)

            df = df[df["winner_proba"] > 0.95]
            predicted_winning_trades = len(df)
            winning_trades = df['win_verdict'].sum() 
            percent = (winning_trades / predicted_winning_trades * 100.0) if predicted_winning_trades >0 else 0.0
            stats.append({
                'file': file.split('/')[-1],
                'predicted_winning_trades': predicted_winning_trades,
                'winning_trades': winning_trades,
                'winning_percentage': percent
            })
        
        stats_df = pd.DataFrame(stats)
        stats_file = os.path.join(labeled_output_dir, "labeled_files_stats.csv")
        stats_df.to_csv(stats_file, index=False)
        print(f"Stats saved to {stats_file}")


    def fill_expiry_close(self, file, filter_weekly=True):
        """
        Function to fill expiry_close column for a given file, and save to labeled folder.
        """
        labled_output_dir = os.getenv("LABELED_OUTPUT_DIR", "prod/output/labeled")
        df = pd.read_csv(file)

        # compensate the expiry_close column
        if 'expirationDate' not in df.columns:
            df['expirationDate'] = df['symbol'].apply(
                lambda s: str(s).upper()[str(s).upper().find('|')+1:str(s).upper().find('|')+9]).apply(
                    lambda d: d[:4] + '-' + d[4:6] + '-' + d[6:])
        


        syms = df['baseSymbol'].dropna().astype(str).str.upper().unique().tolist()

        if 'tradeTime' not in df.columns:
            file_name = file.split('/')[-1]
            date_segment, _ = self.get_date_seg(file_name)
            #tt = pd.to_datetime(date_segment)
            df['tradeTime'] = date_segment
        tt = pd.to_datetime(df.get('tradeTime', pd.NaT), errors="coerce")
        ed = pd.to_datetime(df.get('expirationDate', pd.NaT), errors="coerce")


        # filter for weekly options
        if filter_weekly:
            # only keep weekly options (daysToExpiration <=7)
            if 'daysToExpiration' not in df.columns:
                df['daysToExpiration'] = (ed - tt).dt.days
            df = df[df['daysToExpiration'] <= 7].copy()


        cache_dir = os.getenv("COMMON_OUTPUT_DIR", "./output")
        batch_size = int(os.getenv("DATA_BATCH_SIZE", "30"))
        cut_off_date =  pd.Timestamp.now()

        preload_closes = preload_prices_with_cache(syms, tt, ed, cache_dir, batch_size=batch_size, cut_off_date=cut_off_date)

        df['expiry_close'] = df.apply(lambda r: expiry_close_from_cache(r, preload_closes), axis=1)
        output_file = f"{labled_output_dir}/{file.split('/')[-1]}"
        df.to_csv(output_file, index=False)


if __name__ == '__main__':
    unittest.main()
