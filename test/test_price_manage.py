import sys
import unittest

import pandas as pd
from service.data_prepare import _load_cached_price_data
from service.stock_data_manager2 import GroupedStockUpdater

class TestPriceData(unittest.TestCase):

    def setUp(self):
        pass

    def test_grouping(self):
        # Load the data
        file = "data/price_cache_dates.csv" 
        df = pd.read_csv(file)
        #df['end_date'] = pd.to_datetime(df['end_date'])
        groups = df.groupby('end_date')
        # Check that the groups are formed correctly
        for end_date, group in groups:
            print(f"End Date: {end_date}, Symbols: {group['symbol'].tolist()}")
        self.assertTrue(len(groups) > 0)

    def test_locate_date(self):
        # Load the data
        file = "data/price_cache_dates.csv" 
        df = pd.read_csv(file)
        symbol = "AAPL"
        symbols = ['ATYR', 'LDI', 'PSKY', 'RH', 'CVNA', 'SNDK', 'SWBI', 'SATS', 'ARM', 'M', 'TER', 'GLNG', 'ARCC', 'EQNR', 'GE', 'AGNC', 'JCI', 'COST', 'WOLF', 'GPRO', 'USAR', 'HTZ', 'LUNR', 'FDX', 'COMP', 'DUOL', 'OUST', 'CNC', 'GEO', 'PCG', 'DHI', 'OKTA', 'ABNB', 'HD', 'EFX', 'CPRT', 'KR', 'INOD', 'KC', 'NVAX', 'GIS', 'ADMA', 'HUM', 'ASAN', 'SCHW', 'CARR', 'UNP', 'HPQ', 'RIO', 'CAH', 'ABTC', 'VFC', 'BBY', 'GAP', 'GH', 'BX', 'GSK', 'NEE', 'LEN']
        for symbol in symbols:
        # Locate the row for the symbol
            end_date = self.get_enddate(df, symbol)
            print(f"Symbol: {symbol}, End Date: {end_date}")
            self.assertIsNotNone(end_date)

    def get_enddate(self, df, symbol):
        row = df[df['symbol'] == symbol]
        # get the end date
        if not row.empty:
            end_date = row['end_date'].values[0]
        return end_date
    
    def test_grouped_update(self):
        log_file = "data/price_cache_dates.csv"
        data_folder ="output/price_cache"
        # Create an instance of GroupedStockUpdater
        updater = GroupedStockUpdater(data_dir=data_folder, log_file=log_file)
        # Define a list of symbols to update
        symbols = ['BBBY', 'FIG', 'OXM', 'QSR', 'BLSH', 'SVM', 'ASH', 'FITB', 'LYV', 'TRIN', 'WBTN', 'AACT', 'FLUX', 'HUN', 'DOC', 'FDS', 'ALKS', 'COLD', 'EFC', 'ARR', 'BRK.B', 'TSEM', 'CTRA', 'BSY', 'PLYM', 'IQ', 'SSYS', 'EVLV', 'FLY', 'XRAY']
        # Perform the batch update
        updater.update_batch(symbols)
        # Check that the log file has been updated
        self.assertTrue(os.path.exists('test_stock_log.csv'))
        log_df = pd.read_csv('test_stock_log.csv')
        print(log_df)
        self.assertTrue(not log_df.empty)
    
    def test_check_labeled_data(self):
        #labeled_file = "output/labeled_trades_t1.csv"
        #labeled_file = "output/labeled_trades_tr2.csv"
        labeled_file = "output/labeled_trades_tr3.csv"
        self.assertTrue(os.path.exists(labeled_file))
        labeled_df = pd.read_csv(labeled_file)
        print(labeled_df.head())
        ec = labeled_df['expiry_close']
        # check missing
        missing_count = ec.isna().sum()
        print(f"Missing expiry_close count: {missing_count}")
        labeled_df['tradeTime'] = pd.to_datetime(labeled_df['tradeTime'])
        min_date = labeled_df['tradeTime'].min()
        max_date = labeled_df['tradeTime'].max()
        print(f"Min Date: {min_date}, Max Date: {max_date}")
        self.assertTrue(not labeled_df.empty)

    def test_modify_labeled_data(self):
        labeled_file = "output/labeled_trades_tr3.csv"
        self.assertTrue(os.path.exists(labeled_file))
        labeled_df = pd.read_csv(labeled_file)
        print(labeled_df.head())
        # filter out the rows with "tradeTime" equal to "2025-09-05"
        filtered_df = labeled_df[labeled_df['tradeTime'] != "2025-09-05"]
        print(filtered_df)
        lenght_orig = len(labeled_df)
        lenght_filt = len(filtered_df)
        print(f"Original length: {lenght_orig}, Filtered length: {lenght_filt}")
        # save to new file
        filtered_df.to_csv("output/labeled_trades_b_0906-0911.csv", index=False)

        self.assertTrue(not labeled_df.empty)
    
    def test_group_update_prices(self):
        log_file = "data/price_cache_dates.csv"
        data_folder ="output/price_cache"
        # Create an instance of GroupedStockUpdater
        updater = GroupedStockUpdater(data_dir=data_folder, log_file=log_file)
        # Define a list of symbols to update
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        # Perform the batch update
        updater.update_batch(symbols)
        # Check that the log file has been updated
        self.assertTrue(os.path.exists('test_stock_log.csv'))
        log_df = pd.read_csv('test_stock_log.csv')
        print(log_df)
        self.assertTrue(not log_df.empty)

    def test_fake_wolf_parquet(self):
        print("Testing reading WOLF parquet file")
        wolf_file = "data/wolf/WOLF.csv"
        df = pd.read_csv(wolf_file, index_col="Date")
        real_wolf_parque = "output/price_cache/WOLF.parquet"
        df_parquet = pd.read_parquet(real_wolf_parque)
        print(df_parquet.head())

        # make the df index the datetime type
        df.index = pd.to_datetime(df.index)
        # df rename the column "Vol." to "Volume"
        df.rename(columns={"Vol.": "Volume"}, inplace=True)
        # the volumne column convert to numeric, remove "M" instead of multiply by 1000000
        df["Volume"] = pd.to_numeric(df["Volume"].str.replace("M", ""), errors='coerce')
        df["Volume"] = df["Volume"] * 1000000
        # sort the df by index
        df.sort_index(inplace=True)
        # cut the date to 2025-09-28
        df = df.loc[:'2025-09-28']
        df.to_parquet("data/wolf/WOLF.parquet")

        print(df.head())
    
    def test_get_price(self):
        cache_dir ="output/price_cache"
        #symbol = "GLXY"
        symbol = "BMNR"
        date = "2025-09-29"
        price_df, _ = _load_cached_price_data(cache_dir, symbol)

        print(f"Price of {symbol} on {date} is {price_df.loc[date]}")

    def test_direct_download_price(self):
        import yfinance as yf
        cache_dir ="output/price_cache"
        #symbol = "GLXY"
        symbols = ["BMNR", "BRK-B", "CRCL", "GLXY"]
        date = "2025-09-29"
        start_dt = pd.to_datetime("2025-04-13")
        end_dt = pd.to_datetime("2025-09-30")
        df = yf.download(symbols, start=start_dt.date(), end=(end_dt + pd.Timedelta(days=1)).date())
        for symbol in symbols:
            df_symbol = df.xs(symbol, level=1, axis=1)
            print(f"Price of {symbol} on {date} is {df_symbol['Close'].loc[date]}")

        #print(f"Price of {symbols} on {date} is {df.loc[date]}")

if __name__ == '__main__':
    unittest.main()
