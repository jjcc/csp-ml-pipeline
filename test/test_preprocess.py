import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import tempfile
import os
from datetime import time

from pyparsing import Path
from service.preprocess import merge_gex, pick_daily_snapshot_files
from dotenv import load_dotenv
import json

load_dotenv()

#out_dir = os.getenv("OUT_DIR", "output")
out_dir = "test/data/output"
#csv_file = "coveredPut_2025-08-13_11_00.csv"
csv_file = "coveredPut_2025-06-12_11_00.csv"
csv_postfix = csv_file.replace("coveredPut_","").replace(".csv","")
csv_path = f"option/put/{csv_file}"
#out_path = Path(args.out)
out_path = os.getenv("LABELED_TRADES_WITH_GEX")
out_path = f"{out_dir}/{out_path}_{csv_postfix}"
base_dir = os.getenv("GEX_BASE_DIR")


def parse_target_time(s: str) -> time:
    try:
        hh, mm = s.split(":")
        return time(int(hh), int(mm))
    except Exception:
        return time(11, 0)

class TestPreprocess(unittest.TestCase):
    def setUp(self):
        # Create a temporary CSV file to simulate labeled_trades.csv
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = csv_path

        # Create a dummy GEX base dir (empty for now)
        self.gex_base_dir =  base_dir
        self.base_dir = base_dir

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_merge_gex_runs_and_returns_dataframe(self):
        # Should run and return a DataFrame, even if GEX files are missing
        target_time_str = "11:00"
        target_t = parse_target_time(target_time_str)
        target_minutes = target_t.hour * 60 + target_t.minute

        self.csv_path = "test/data/put/coveredPut_test.csv"
        trades = pd.read_csv(self.csv_path)
        merged = merge_gex(trades, self.gex_base_dir, target_minutes=target_minutes)  # 11:00am = 660 minutes
        out_path = f"{out_dir}/merged_test.csv"
        merged.to_csv(out_path, index=False)


        self.assertIsInstance(merged, pd.DataFrame)
        self.assertIn('gex_missing', merged.columns)

    def test_get_option_by_date(self):
        # Test getting option data by date
        target_date = "2025-08-07"
        target_time = "11:00"
        data_dir = os.getenv("DATA_DIR", "")
        #glob_pat = getenv("GLOB", "coveredPut_*.csv")

        glob_pat = f"coveredPut_{target_date}*.csv"

        #paths = pick_daily_snapshot_files(data_dir, glob_pat, target_time)
        paths = sorted([str(Path(p)) for p in Path(data_dir).glob(glob_pat)])
        print(f"Paths found for {target_date}: {paths}")
        #option_data = get_option_by_date(self.csv_path, target_date)
        self.assertGreater(len(paths), 0)

    def get_snap_shot(self, target_date="2025-08-13", target_time="11:00"): 
        #option_data_dir = "test/data/put"
        option_data_dir = "option/put"
        glob_pat = f"coveredPut_{target_date}*.csv"
        paths = pick_daily_snapshot_files(option_data_dir, glob_pat, target_time)
        return paths
    
    def test_pick_daily_snapshot_files(self):
        paths = self.get_snap_shot()
        print(f"Paths found  {paths}")
        self.assertGreater(len(paths), 0)
    
    def test_merge_gex_one(self):
        target_date = "2025-08-08"
        target_time = "11:00"
        target_t = parse_target_time(target_time)
        target_minutes = target_t.hour * 60 + target_t.minute
        paths = self.get_snap_shot(target_date=target_date, target_time=target_time)
        option_file = paths[0]

        out_path = f"{out_dir}/merged_test2_{target_date}.csv"
        gex_base_dir = self.gex_base_dir
        merged = self.merge_option_ges(target_minutes, gex_base_dir, option_file, out_path=out_path)
        self.assertIsInstance(merged, pd.DataFrame, merged)

    def merge_option_ges(self,target_minutes, gex_base_dir, option_file, out_path):
        trades = pd.read_csv(option_file)
        merged = merge_gex(trades, gex_base_dir, target_minutes=target_minutes)
        # got "merged_test_2025-08-13.csv"
        merged.to_csv(out_path, index=False)
        return merged



if __name__ == '__main__':
    unittest.main()
