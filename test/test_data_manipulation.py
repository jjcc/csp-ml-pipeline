import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# training dataset: Apr 25 to Aug 08
# test dataset a: Aug 11 to Sep 05
# test dataset b: Sep 08 to Sep 26

class TestDataManipulation(unittest.TestCase):

    def test_merge_tr_test1_dataframes(self):
        '''
        Merge original training dataset with the first test dataset
        '''
        file1 = "output/labeled_trades.csv"
        file2 = "output/labeled_trades_t2.csv"
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        df_merged = pd.concat([df1, df2]).drop_duplicates().reset_index(drop=True)
        print(f"Merged DataFrame has {len(df_merged)} rows.")
        df_merged.to_csv("output/labeled_trades_tr_t1_merged.csv", index=False)

    def test_merge_tr_test1_dataframes_minus(self):
        '''
        Merge original training dataset with the first test dataset
        Remove the last N rows from the original training dataset before merging, to keep the size the same
        '''
        file1 = "output/labeled_trades.csv"
        file2 = "output/labeled_trades_t2.csv"
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        length_test = len(df2)
        df1 = df1.iloc[:-length_test]
        df_merged = pd.concat([df1, df2]).drop_duplicates().reset_index(drop=True)
        print(f"Merged DataFrame has {len(df_merged)} rows.")
        df_merged.to_csv("output/labeled_trades_tr_t1_merged_minus.csv", index=False)
    
    def test_merge_tr_with_AB(self):
        '''
        Merge original training dataset with the test datasets A and B
        '''
        file1 = "output/labeled_trades.csv"
        file2 = "output/trades_with_gex_macro_a_0811fix.csv"
        file3 = "output/trades_with_gex_macro_B_0908.csv"
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        df3 = pd.read_csv(file3)
        df_merged = pd.concat([df1, df2, df3]).drop_duplicates().reset_index(drop=True)
        print(f"Merged DataFrame has {len(df_merged)} rows.")
        df_merged.to_csv("output/labeled_trades_tr_A_B_merged.csv", index=False)

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

    def test_fix_dataset_a(self):
        '''
        Fix original dataset a by removing rows with tradeTime on 2025-08-08
        '''
        #file1 = "output/labeled_trades_a_0808-0905_old.csv"
        #file1fix = "output/labeled_trades_a_0808-0905_fixed.csv"
        #file1 = "output/trades_raw_t2.csv"
        #file1fix = "output/trades_raw_t2_fixed.csv"
        file1 = "output/trades_with_gex_macro_a_0811.csv"
        file1fix = "output/trades_with_gex_macro_a_0811fix.csv"
        
        df1 = pd.read_csv(file1)
        print(f"Original DataFrame has {len(df1)} rows.")
        df1['tradeTime'] = pd.to_datetime(df1['tradeTime'], errors='coerce')
        df_fixed = df1[df1['tradeTime'] != pd.Timestamp("2025-08-08")]
        print(f"Fixed DataFrame has {len(df_fixed)} rows.")
        df_fixed.to_csv(file1fix, index=False)


    def test_cutoff_date(self):
        from service.env_config import getenv
        specifics = ["aug_11", "sep_8", "sep_12"] #  "sep_29" is test
        print("\nConfiguration cutoff dates:")
        for spec in specifics:
            basic_csv = getenv(f"COMMON_CONFIGS_{spec.upper()}_DATA_BASIC_CSV", "")
            #print(f"COMMON_CONFIGS_{spec.upper()}_DATA_BASIC_CSV: {basic_csv}")
            df = pd.read_csv(f"output/{basic_csv}")
            df['expirationDate'] = pd.to_datetime(df['expirationDate'], errors='coerce')
            min_date = df['expirationDate'].min()
            print(f"  Min expirationDate in {basic_csv}: {min_date}")
        #self.assertEqual(cutoff_date, "2025-08-11")
if __name__ == '__main__':
    unittest.main()
