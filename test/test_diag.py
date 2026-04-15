import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from service.utils import prep_tail_training_df, fill_features_with_training_medians
from service.utils import ALL_FEATS
from train_tail_with_gex import _fill_features
from dotenv import load_dotenv
load_dotenv()

class TestDiag(unittest.TestCase):

    def setUp(self):
        self.file1 = "output/winner_score/scores_winner_v6_oof_ne_ts_w_lgbm.csv"
        self.file2 = "prod/output/scored_tail_winner_test_2025-08-08_11_00_test.csv"
        self.df1 = pd.read_csv(self.file1)
        self.df2 = pd.read_csv(self.file2)
        self.model = joblib.load("output/winner_train/v6_oof_ne_straited_w_lgbm/winner_classifier_v6_oof_ne_w_lgbm.pkl")

        pass

    def test_verify_features(self):
        """
        """
        feats = self.model.get("features", None)
        mediams = self.model.get("medians", None)
        print(f"Model features ({len(feats)}): {feats}")
        col1 = set(self.df1.columns)
        col2 = set(self.df2.columns)
        missing1 = [f for f in feats if f not in col1]
        missing2 = [f for f in feats if f not in col2]
        self.assertTrue(len(missing1) == 0, f"Missing features in {self.file1}: {missing1}")
        self.assertTrue(len(missing2) == 0, f"Missing features in {self.file2}: {missing2}")


        


    def test_verify_feats(self):
        feats = self.model.get("features", None)
        mediams = self.model.get("medians", None)
        for index, row in self.df2.iterrows():
            row1 = self.df1.iloc[index]
            for feat in feats:
                val2 = row.get(feat, None)
                val1 = row1.get(feat, None)
                val1a = self.df1.loc[index, feat]
                if val1 != val2:
                    print(f"Row {index} feature {feat} differs: {val1} (file1) vs {val2} (file2)")  
                #self.assertEqual(val1, val2, f"Value mismatch for feature {feat} in row {index}")

    def test_verify_prediction(self):
        for index, row in self.df2.iterrows():
            row1 = self.df1.iloc[index]
            prob1 = row1.get("win_proba", None)
            prob2 = row.get("winner_proba", None)
            if prob1 != prob2:
                print(f"Row {index} prediction differs: {prob1} (file1) vs {prob2} (file2)")
                #self.assertEqual(val1, val2, f"Value mismatch for feature {feat} in row {index}")

if __name__ == '__main__':
    unittest.main()
