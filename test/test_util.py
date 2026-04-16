import sys
import unittest
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from service.utils import prep_tail_training_df, fill_features_with_training_medians
from service.utils import ALL_FEATS

class TestUtil(unittest.TestCase):

    def test_prep_tail_training_derived(self):
        """
        Compare prep_tail_training_derived and _prep_df outputs
        """
        df = pd.read_csv("output/labeled_trades_with_gex_normal.csv")
        result1 = prep_tail_training_df(df)
        #result2 = _prep_df(df)
        # Check if both functions produce the same result
        #pd.testing.assert_frame_equal(result1, result2)

        

        #
        MODEL_IN = "models/tail_model_gex_v2_cut05.pkl"
        pack = joblib.load(MODEL_IN)
        med  = pack["medians"]
        score_x, medians_x = fill_features_with_training_medians(result1, ALL_FEATS)
        #score_t, medians_t = _fill_features(result2, ALL_FEATS)

        #pd.testing.assert_frame_equal(score_x, score_t)
        #assert(medians_x == medians_t)

        self.assertIsInstance(result1, pd.DataFrame)
    
    def test_compare_different_datasets(self):
        """
        compare the columns of basic, gex, and macro feature datasets
        """
        df_basic = pd.read_csv("output/labeled_trades_normal.csv")
        df_gex = pd.read_csv("output/labeled_trades_with_gex_normal.csv")
        df_macro = pd.read_csv("output/labeled_trades_normal_gex_macro.csv")

        cols_basic = set(df_basic.columns)
        cols_gex = set(df_gex.columns)
        cols_macro = set(df_macro.columns)

        gex_extra = cols_gex - cols_basic
        macro_extra = cols_macro - cols_gex
        print(f"gex extra = {gex_extra}")
        print(f"macro extra = {macro_extra}")

        # GEX dataset should have all columns of basic dataset plus some additional columns
        self.assertTrue(cols_basic.issubset(cols_gex))
        self.assertTrue(len(cols_gex) > len(cols_basic))

        # Macro dataset should have all columns of GEX dataset plus some additional columns
        self.assertTrue(cols_gex.issubset(cols_macro))
        self.assertTrue(len(cols_macro) > len(cols_gex))

        #gex extra = {'gex_pos', 'gex_missing', 'gex_distance_to_flip', 'gex_total_abs', 'symbol_norm', 'gex_sign_at_ul', 'gex_flip_strike', 'gex_file', 'trade_date', 'gex_neg', 'gex_center_abs_strike', 'gex_total', 'gex_gamma_at_ul'}
        #macro extra = {'VIX', 'ret_5d_norm', 'prev_close_minus_strike', 'ret_2d', 'prev_close_minus_strike_pct', 'log1p_DTE', 'prev_close', 'ret_5d', 'ret_2d_norm'}


if __name__ == '__main__':
    unittest.main()
