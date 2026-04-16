#!/usr/bin/env python3

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unittest import skip 
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from unittest.mock import patch

# Import the function to test
import sys
from service.data_prepare import _load_vix
from service.get_vix import init_driver, url_vix
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestLoadVix(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.start_date = "2025-04-01"

        self.end_date = "2025-10-03"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    

    
    def test_get_current_vix(self):
        """Test fetching current VIX value from the web page"""
        from service.get_vix import get_current_vix
        driver = init_driver(headless=True)


        vix_value = get_current_vix(url_vix, driver)
        print("Fetched VIX value:", vix_value)
        
        self.assertIsNotNone(vix_value)
        try:
            vix_float = float(vix_value)
            self.assertGreater(vix_float, 0)
        except ValueError:
            self.fail("VIX value is not a valid float")
    
    def test_load_from_csv_file_with_Date_Close_columns(self):
        """Test loading VIX from CSV file with Date/Close columns"""
        # Create test CSV file
        VIX_CSV     = os.getenv("VIX_CSV", "").strip() or None
        csv_path = VIX_CSV
        
        end_date = pd.to_datetime(self.end_date) - pd.Timedelta(days=1)
        result = _load_vix(str(csv_path), self.start_date, end_date, force_download=True)

        self.assertEqual(result.name, "VIX")
        self.assertGreater(len(result), 5)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result.index))
    

    
    def test_csv_file_not_exists(self):
        """Test when CSV file does not exist and use_yf=False"""
        non_existent_path = Path(self.temp_dir) / "nonexistent.csv"
        
        result = _load_vix(str(non_existent_path), self.start_date, self.end_date, use_yf=False)
        
        self.assertEqual(result.name, "VIX")
        self.assertTrue(result.empty)
        self.assertEqual(result.dtype, float)
    
    def test_none_vix_csv_path(self):
        """Test when vix_csv is None and use_yf=False"""
        result = _load_vix(None, self.start_date, self.end_date, use_yf=False)
        
        self.assertEqual(result.name, "VIX")
        self.assertTrue(result.empty)
        self.assertEqual(result.dtype, float)
    
    def test_empty_vix_csv_path(self):
        """Test when vix_csv is empty string and use_yf=False"""
        result = _load_vix("", self.start_date, self.end_date, use_yf=False)
        
        self.assertEqual(result.name, "VIX")
        self.assertTrue(result.empty)
        self.assertEqual(result.dtype, float)
    
    @skip("Ignore this test")
    def test_date_filtering(self):
        """Test that date filtering works correctly"""
        # Create test CSV with dates outside the range
        test_data = pd.DataFrame({
            'Date': ['2022-12-30', '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-15'],
            'Close': [15.0, 20.5, 21.0, 19.8, 25.0]
        })
        csv_path = Path(self.temp_dir) / "vix_filter.csv"
        test_data.to_csv(csv_path, index=False)
        
        result = _load_vix(str(csv_path), "2023-01-01", "2023-01-03", use_yf=False)
        
        self.assertEqual(len(result), 3)  # Should only include dates in range
        self.assertAlmostEqual(result.iloc[0], 20.5)  # First should be 2023-01-01
        self.assertAlmostEqual(result.iloc[-1], 19.8)  # Last should be 2023-01-03

    @skip("Ignore this test")
    def test_invalid_dates_handling(self):
        """Test handling of invalid dates in CSV"""
        test_data = pd.DataFrame({
            'Date': ['2023-01-01', 'invalid-date', '2023-01-03'],
            'Close': [20.5, 21.0, 19.8]
        })
        csv_path = Path(self.temp_dir) / "vix_invalid.csv"
        test_data.to_csv(csv_path, index=False)
        
        result = _load_vix(str(csv_path), self.start_date, self.end_date, use_yf=False)
        
        # Should only include valid dates (invalid row should be dropped)
        self.assertEqual(len(result), 2)
    
    def test_yfinance_fallback_success(self):
        """Test successful yfinance fallback when CSV not available"""
        import yfinance as yf
        # Mock yfinance download
        #mock_data = pd.DataFrame({
        #    'Close': [20.5, 21.0, 19.8]
        #}, index=pd.date_range('2023-01-01', periods=3))
        #mock_yf.download.return_value = mock_data

        VIX_CSV="output/vix_data.csv"
        result = _load_vix(VIX_CSV, self.start_date, self.end_date)
        
        #mock_yf.download.assert_called_once_with("^VIX", start=self.start_date, end=self.end_date)
        self.assertEqual(result.name, "VIX")
        self.assertGreater(len(result), 0)
    
    

if __name__ == '__main__':
    unittest.main()