import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd


round_digits = 6 

class TestRefactorVerification(unittest.TestCase):
    """
    Test class to verify refactored code for data preparation."""


    def test_regulate_float(self):
        """
        change the float precision to avoid small differences in float calculations
        This function reads two CSV files, rounds specific float columns to a defined number of decimal places,
        and saves the modified DataFrames back to new CSV files.
        """
        file_o = 'test/data/trades_with_gex_macro_a_0811.csv'
        file_t1 = 'test/data/trades_with_gex_macro_a_0811_t1.csv'

        df = pd.read_csv(file_o)
        df_t1 = pd.read_csv(file_t1)
        cols = df.columns.tolist() 
        cols_to_check = [col for col in cols if re.match(r'^(gex_).+', col)]
        cols_to_check  = [col for col in cols_to_check if col not in ['gex_file', 'gex_missing','gex_sign_at_ul']] # remove non-float columns

        for col in cols_to_check:
            df[col] = df[col].round(round_digits)
            df_t1[col] = df_t1[col].round(round_digits)
        
        # save the file with new name
        df.to_csv('test/data/trades_with_gex_macro_a_0811_regulated.csv', index=False)
        df_t1.to_csv('test/data/trades_with_gex_macro_a_0811_t1_regulated.csv', index=False)
        # use winmerge or other diff tool to compare the two files


    def test_compare_files(self):
        """
        Compare two CSV files to check if they are identical after float regulation.
        """
        file_1 = 'test/data/trades_with_gex_macro_a_0811_regulated.csv'
        file_2 = 'test/data/trades_with_gex_macro_a_0811_t1_regulated.csv'

        df1 = pd.read_csv(file_1)
        df2 = pd.read_csv(file_2)

        # go over 2 dataframes and compare row by row, record the row with differences, collect index and symbol
        for index, row in df1.iterrows():
            if not row.equals(df2.iloc[index]):
                idx = index
                symbol = row['baseSymbol'] 
                print(f"Difference found at index {index}, symbol {symbol}")










if __name__ == '__main__':
    unittest.main()
