import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import os

from dotenv import load_dotenv

from service.option_metrics import compute_option_metrics

load_dotenv()


class TestOptionMetrics(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass


    def test_get_metrics(self):
        option_data_dir = "option/put"
        glob_pat = f"coveredPut_*.csv"
        # get the latest file
        latest_file_with_path = max(glob(os.path.join(option_data_dir, glob_pat)), key=os.path.getctime)
        a_file = latest_file_with_path

        df_o = pd.read_csv(a_file)
        df = df_o[["underlyingLastPrice", "strike", "bidPrice", "daysToExpiration"]].copy()

        out = compute_option_metrics(df, iv_col="impliedVolatility", add_probabilities=False)

        # Inspect the enriched DataFrame
        print(out.head())
        count = 0
        for idx, row in out.iterrows():
            row_o = df_o.iloc[idx]
            for col in row.index:
                if col in row_o.index:
                    diff =  row[col] - row_o[col]
                    if diff > 0.0001:
                        print(f"Significant difference in row {idx}, column '{col}': cal:{row[col]} != orig:{row_o[col]}")
                        count += 1
        print(f"Total significant differences found: {count}")

        out.to_csv("test/data/output/with_metrics.csv", index=False)

 

if __name__ == '__main__':
    unittest.main()
