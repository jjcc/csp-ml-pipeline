import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import joblib
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import time

from pyparsing import Path
from dotenv import load_dotenv
import json

from service.preprocess import merge_gex, pick_daily_snapshot_files
from service.utils import fill_features_with_training_medians, prep_tail_training_df, prep_winner_like_training

load_dotenv()

#out_dir = os.getenv("OUT_DIR", "output")
out_dir = "test/data/output"
#csv_file = "coveredPut_2025-08-13_11_00.csv"
csv_file = "coveredPut_2025-06-12_11_00.csv"
csv_postfix = csv_file.replace("coveredPut_","").replace(".csv","")
csv_path = f"test/data/put/{csv_file}"
#out_path = Path(args.out)
out_path = os.getenv("LABELED_TRADES_WITH_GEX")
out_path = f"{out_dir}/{out_path}_{csv_postfix}"
base_dir = os.getenv("GEX_BASE_DIR")


TAIL_MODEL_IN="models/tail_model_gex_v2_cut05.pkl"
TAIL_KEEP_PROBA_COL="tail_proba"
WINNER_MODEL_IN="output/winner/model_pack.pkl"
WINNER_PROBA_COL = "proba"

def parse_target_time(s: str) -> time:
    try:
        hh, mm = s.split(":")
        return time(int(hh), int(mm))
    except Exception:
        return time(11, 0)

class TestProd(unittest.TestCase):
    def setUp(self):
        # Create a temporary CSV file to simulate labeled_trades.csv
        self.csv_path = csv_path
        # Create a dummy GEX base dir (empty for now)
        self.gex_base_dir =  base_dir
        self.base_dir = base_dir

    def tearDown(self):
        pass

    def test_merge_score_both(self):
        pass

    def get_snap_shot(self, target_date="2025-08-13", target_time="11:00"): 
        #option_data_dir = "test/data/put"
        option_data_dir = "option/put"
        glob_pat = f"coveredPut_{target_date}*.csv"
        paths = pick_daily_snapshot_files(option_data_dir, glob_pat, target_time)
        return paths

    def test_scan_latest(self):
        latest_file_with_path, target_minutes = self.get_merge_params()
        if latest_file_with_path is None:
            return
        gex_base_dir = os.getenv("GEX_BASE_DIR")

        trades = pd.read_csv(latest_file_with_path)
        df_o = merge_gex(trades, gex_base_dir, target_minutes=target_minutes)
        print(df_o.head())

        #with open(process_log, "a") as f:
        #    f.write(f"{latest_file}\n")
        #print(f"Latest file: {latest_file}")
        # get latest snapshot files
        #paths = self.get_snap_shot()
        #self.assertTrue(len(paths) > 0)

    def get_merge_params(self):
        option_data_dir = "option/put"
        glob_pat = f"coveredPut_*.csv"
        # get the latest file
        latest_file_with_path = max(glob(os.path.join(option_data_dir, glob_pat)), key=os.path.getctime)
        # file only
        latest_file = os.path.basename(latest_file_with_path)
        process_log = "log/processed.log"
        with open(process_log, "r") as f:
            lines = f.readlines()
        files = [line.strip() for line in lines if line.strip()]
        if latest_file in files:
            print(f"Latest file {latest_file} already processed.")
            return None, None
        # process the latest file
        # get the time of latest_file, like coveredPut_2025-08-15_16_30.csv
        latest_file_time = latest_file.split("_")[-2:]
        latest_file_time = ":".join(latest_file_time).replace(".csv", "")
        target_t = parse_target_time(latest_file_time)
        target_minutes = target_t.hour * 60 + target_t.minute
        return latest_file_with_path,target_minutes


    # Test both
    def test_score_tail_winner_classifier(self):
        #target_date = "2025-08-08"
        #target_time = "11:00"
        #time_str = target_time.replace(":", "_")

        #target_t = parse_target_time(target_time)
        #target_minutes = target_t.hour * 60 + target_t.minute

        #paths = self.get_snap_shot(target_date=target_date, target_time=target_time)
        #option_file = paths[0]
        latest_file_with_path, target_minutes = self.get_merge_params()
        if latest_file_with_path is None:
            print("File is processed.")
            return

        option_file = latest_file_with_path

        # construct a output file
        hour, minute = option_file.replace(".csv", "").split("_")[-2:]
        target_date = option_file.split("_")[1]
        time_str = f"{hour}_{minute}"
        out_path = f"prod/output/scored_tail_winner_test_{target_date}_{time_str}.csv"

        gex_base_dir = os.getenv("GEX_BASE_DIR")
        trades = pd.read_csv(option_file)
        df_o = merge_gex(trades, gex_base_dir, target_minutes=target_minutes)

        pack_tl = joblib.load(TAIL_MODEL_IN)
        clf_tl = pack_tl["model"]
        feats = pack_tl["features"]
        med  = pack_tl["medians"]
        df = prep_tail_training_df(df_o)
        X_tl, medians = fill_features_with_training_medians(df, feats)

        proba = clf_tl.predict_proba(X_tl)[:,1]
        out = df.copy()
        out[TAIL_KEEP_PROBA_COL] = proba
        thresh = 0.04028
        out["is_tail_pred"] = (out[TAIL_KEEP_PROBA_COL] >= thresh).astype(int)


        pack_wc = joblib.load(WINNER_MODEL_IN)
        clf_wc = pack_wc["model"]
        feats = pack_wc["features"]
        medians = pack_wc.get("medians", None)
        impute_missing = bool(pack_wc.get("impute_missing", bool(medians is not None)))
        Xwc, mask = prep_winner_like_training(df, feats, medians=medians, impute_missing=impute_missing)

        proba = clf_wc.predict_proba(Xwc)[:,1]
        out2 = out.loc[mask].copy()
        out2[WINNER_PROBA_COL] = proba
        thresh = 0.8375
        out2["is_winner_pred"] = (out2[WINNER_PROBA_COL] >= thresh).astype(int)

        gex_columns = [col for col in out2.columns if col.startswith("gex_")]
        out2.drop(columns=gex_columns, inplace=True, errors='ignore')
        to_drop = [ 'baseSymbolType',  'expirationDate',  
            'strike', 'moneyness', 'breakEvenBid', 'percentToBreakEvenBid','tradeTime','symbol_norm','impliedVolatilityRank1y',
            'delta',  'breakEvenProbability', 'expirationType', 'symbolType', 
            'entry_credit', 'exit_intrinsic', 'total_pnl', 'return_pct']
        out2.drop(columns=to_drop, inplace=True, errors='ignore')
        out2 = out2[out2["is_winner_pred"] == 1]
        out2 = out2[out2["is_tail_pred"] == 0]
        out2.to_csv(out_path, index=False)
        # log processing information
        process_log = "log/processed.log"
        latest_file = os.path.basename(latest_file_with_path)
        with open(process_log, "a") as f:
            f.write(f"{latest_file}\n")
        print(f"Latest file: {latest_file}")

if __name__ == '__main__':
    unittest.main()
