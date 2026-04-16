#!/usr/bin/env python3
import sys

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from unittest.mock import patch, MagicMock
import yfinance as yf
import time

# Import the function to test
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service.data_prepare import _load_symbol_prices



def collect_fundamentals(symbols):
    rows = []
    length = len(symbols)
    for idx, sym in enumerate(symbols):
        try:
            info = yf.Ticker(sym).get_info()
            rows.append({
                "symbol": sym,
                "marketcap": info.get("marketCap"),
                "sector": info.get("sector")
            })
            time.sleep(0.5)  # be nice to Yahoo servers
        except Exception as e:
            print(f"Failed {sym}: {e}")
        if idx % 100 == 0:
            print(f"Processed {idx+1}/{length} symbols...")
    return pd.DataFrame(rows)

def collect_earnings(symbols, limit=16):
    rows = []
    length = len(symbols)
    for idx, sym in enumerate(symbols):
        try:
            df = yf.Ticker(sym).get_earnings_dates(limit=limit)
            df = df.reset_index().rename(columns={"Earnings Date": "earnings_date"})
            df["symbol"] = sym
            rows.append(df[["symbol", "earnings_date"]])
            time.sleep(0.5)
        except Exception as e:
            print(f"Failed {sym}: {e}")
        if idx % 100 == 0:
            print(f"Processed {idx+1}/{length} symbols...")
    return pd.concat(rows, ignore_index=True)





class TestLoadSymbolPrices(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.symbol = "AAPL"
        self.start_date = "2025-06-06"
        self.end_date = "2025-08-08"
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_from_csv_file_with_Date_Close_columns(self):
        """Test loading symbol prices from CSV file with Date/Close columns"""
        path = "output/price_cache"
        symbol = "AAPL"
        
        result = _load_symbol_prices(symbol, path, self.start_date, self.end_date, use_yf=False)

        self.assertEqual(result.name, "Close")
        self.assertGreater(len(result), 0)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result.index))
        self.assertGreater(result.iloc[0], 0)
        self.assertGreater(result.iloc[-1], 0)
    
    
    def test_csv_file_not_exists(self):
        """Test when CSV file does not exist and use_yf=False"""
        result = _load_symbol_prices(self.symbol, self.temp_dir, self.start_date, self.end_date, use_yf=False)
        
        self.assertEqual(result.name, "Close")
        self.assertTrue(result.empty)
        self.assertEqual(result.dtype, float)
    
    def test_no_px_dir_provided(self):
        """Test when no px_dir is provided and use_yf=False"""
        result = _load_symbol_prices(self.symbol, None, self.start_date, self.end_date, use_yf=False)
        
        self.assertEqual(result.name, "Close")
        self.assertTrue(result.empty)
        self.assertEqual(result.dtype, float)
    
    def test_date_filtering(self):
        """Test that date filtering works correctly"""
        # Create test CSV with dates outside the range
        test_data = pd.DataFrame({
            'Date': ['2022-12-30', '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-15'],
            'Close': [140.0, 150.0, 151.0, 149.0, 160.0]
        })
        csv_path = Path(self.temp_dir) / f"{self.symbol}.csv"
        test_data.to_csv(csv_path, index=False)
        
        result = _load_symbol_prices(self.symbol, self.temp_dir, "2023-01-01", "2023-01-03", use_yf=False)
        
        self.assertEqual(len(result), 3)  # Should only include dates in range
        self.assertAlmostEqual(result.iloc[0], 150.0)  # First should be 2023-01-01
        self.assertAlmostEqual(result.iloc[-1], 149.0)  # Last should be 2023-01-03



    def get_earnings_calendar(self, symbols, limit=40):
        rows = []
        for sym in symbols:
            try:
                df = yf.Ticker(sym).get_earnings_dates(limit=limit)
                df = df.reset_index().rename(columns={"Earnings Date": "earnings_date"})
                df["symbol"] = sym
                rows.append(df[["symbol", "earnings_date"]])
            except Exception as e:
                print(f"Failed {sym}: {e}")
        return pd.concat(rows, ignore_index=True)
    

    def test_load_calendar_with_yfinance(self):
        """Test loading symbol prices with yfinance"""
        symbols = ["AAPL", "MSFT", "TSLA"]
        earnings_calendar = self.get_earnings_calendar(symbols)
        earnings_calendar.to_csv("test/data/earnings_calendar.csv", index=False)
        assert not earnings_calendar.empty
    
    def test_unique_base_symbol(self):
        file = "output/labeled_trades_normal.csv"
        df = pd.read_csv(file)
        unique_base_symbols = df["baseSymbol"].unique()
        with open("test/data/unique_sym.json","w") as f:
            import json
            json.dump(list(unique_base_symbols), f, indent=2)
        assert len(unique_base_symbols) == df["baseSymbol"].nunique()   

    def test_empty_symbol(self):
        """Test with empty symbol string"""
        result = _load_symbol_prices("", None, self.start_date, self.end_date, use_yf=False)
        
        self.assertEqual(result.name, "Close")
        self.assertTrue(result.empty)
        self.assertEqual(result.dtype, float)
    
    def test_download_cal_fundemental(self):
        with open("test/data/unique_sym.json","r") as f:
            import json
            symbols = json.load(f)
        earnings_calendar = collect_earnings(symbols)
        fundamentals = collect_fundamentals(symbols)

        earnings_calendar.to_csv("output/symbol_info/earnings_calendar.csv", index=False)
        fundamentals.to_csv("output/symbol_info/fundamentals.csv", index=False)
        assert not earnings_calendar.empty
        assert not fundamentals.empty
        print("Download complete!")
    

    def test_verify_cal_earning(self):
        symbols = ["FI","BMNR"]
        earnings_calendar = collect_earnings(symbols)
        assert not earnings_calendar.empty
        print(earnings_calendar)
        assert len(earnings_calendar) > 0


if __name__ == '__main__':
    unittest.main()