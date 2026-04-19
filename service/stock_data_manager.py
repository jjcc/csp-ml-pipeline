import os
import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv

from service.constants import DEFAULT_SLEEP_SECONDS

class GroupedStockUpdater:
    def __init__(self, data_dir='stock_data', log_file='stock_log.csv'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.log_file = Path(log_file)
        self.log_df = self._load_log()

    def _load_log(self):
        """Load the existing log of symbols and their end dates"""
        if self.log_file.exists():
            return pd.read_csv(self.log_file)
        return pd.DataFrame(columns=['symbol', 'end_date'])

    def _save_log(self):
        """Save updated log"""
        self.log_df.to_csv(self.log_file, index=False)

    def update_batch(self, symbols, target_end_date=None):
        """
        Update a batch of symbols by grouping them by their current end_date

        Args:
            symbols: List of stock symbols to update
            target_end_date: Target end date (default: yesterday)
        """
        from service.data_prepare import _load_cached_price_data
        from service.env_config import getenv
        load_dotenv(override=False)
        history_sleep_seconds = float(os.getenv("YF_HISTORY_SLEEP_SECONDS", str(DEFAULT_SLEEP_SECONDS)))
        threads_override = os.getenv("YF_HISTORY_THREADS", "").strip().lower()
        use_threads = threads_override in {"1", "true", "yes", "y", "on"} if threads_override else True
        if target_end_date is None:
            target_end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        # Add 1 day to end_date for yfinance API call
        yf_end_date = (datetime.strptime(target_end_date, '%Y-%m-%d') +
                       timedelta(days=1)).strftime('%Y-%m-%d')

        print(f"Target end date: {target_end_date}")

        # Categorize symbols into groups
        groups = self._categorize_symbols(symbols, target_end_date)

        # Process each group
        total_new = len(groups['new'])
        total_updated = 0

        print(f"\n=== Summary ===")
        print(f"New symbols: {total_new}")
        print(f"Update groups: {len(groups['updates'])}")
        print(f"Already current: {len(groups['current'])}")

        # Need to remove "WOLF" from any group, as it is a special case
        for key in ['new', 'current']:
            if 'WOLF' in groups[key]:
                groups[key].remove('WOLF')
        if 'WOLF' in groups['updates']:
            del groups['updates']['WOLF']

        price_info = {}
        # Download new symbols (full history from April)
        if groups['new']:
            print(f"\n[1/2] Downloading {len(groups['new'])} new symbols...")
            #self._download_group(groups['new'], '2024-04-01', target_end_date)
            res = self._download_group(groups['new'], '2024-04-01', yf_end_date, use_threads=use_threads)
            price_info.update(res)
            if history_sleep_seconds > 0 and groups['updates']:
                time.sleep(history_sleep_seconds)

        # Download updates grouped by end_date
        if groups['updates']:
            print(f"\n[2/2] Updating existing symbols in {len(groups['updates'])} groups...")
            for idx, (end_date, syms) in enumerate(groups['updates'].items(), 1):
                # Calculate start date (day after current end_date)
                start_date = (datetime.strptime(end_date, '%Y-%m-%d') +
                            timedelta(days=1)).strftime('%Y-%m-%d')

                if start_date > target_end_date:
                    continue  # Skip if already up to date

                print(f"  Group {idx}/{len(groups['updates'])}: {len(syms)} symbols "
                      f"({end_date} -> {target_end_date})")

                #self._download_and_append(syms, start_date, target_end_date, end_date)
                prices = self._download_and_append(
                    syms, start_date, yf_end_date, end_date, use_threads=use_threads
                )
                price_info.update(prices)
                total_updated += len(syms)
                if history_sleep_seconds > 0 and idx < len(groups['updates']):
                    time.sleep(history_sleep_seconds)
        if groups['current']:
            out_dir = getenv("COMMON_OUTPUT_DIR", "./output")
            cache_dir = os.path.join(out_dir, "price_cache")
            for s in groups['current']:
                price_df, _ = _load_cached_price_data(cache_dir, s)
                price_info[s] = price_df
            total_updated += len(groups['current'])

        print(f"\n=== Complete ===")
        print(f"Total new: {total_new}")
        print(f"Total updated: {total_updated}")
        print(f"Already current: {len(groups['current'])}")

        return total_new, total_updated , price_info

    def _categorize_symbols(self, symbols, target_end_date):
        """
        Categorize symbols into:
        - new: symbols not in log
        - updates: dict of {end_date: [symbols]} for symbols needing updates
        - current: symbols already at target_end_date
        """
        new_symbols = []
        update_groups = defaultdict(list)
        current_symbols = []

        # Create a lookup dict for faster access
        log_dict = dict(zip(self.log_df['symbol'], self.log_df['end_date']))

        for symbol in symbols:
            if symbol not in log_dict:
                new_symbols.append(symbol)
            elif log_dict[symbol] < target_end_date:
                # Group by current end_date
                update_groups[log_dict[symbol]].append(symbol)
            else:
                current_symbols.append(symbol)

        return {
            'new': new_symbols,
            'updates': dict(update_groups),
            'current': current_symbols
        }

    def _download_group(self, symbols, start_date, end_date, use_threads=True):
        """Download full history for a group of symbols"""
        try:
            # Batch download (much faster)
            data = yf.download(symbols, start=start_date, end=end_date,
                             group_by='ticker', threads=use_threads, progress=True)
            price_info = {}
            # Save each symbol
            for symbol in symbols:
                try:
                    if len(symbols) == 1:
                        df = data
                    else:
                        df = data[symbol]

                    if df.empty:
                        print(f"  Warning: No data for {symbol}")
                        continue

                    # Save to parquet
                    file_path = self.data_dir / f"{symbol}.parquet"
                    df.to_parquet(file_path)
                    price_info[symbol] = df
                    print(f"  Saved {symbol} data with {len(df)} rows.")

                    # Update log
                    self._update_log_entry(symbol, start_date, end_date)

                except Exception as e:
                    print(f"  Error saving {symbol}: {e}")

            self._save_log()
            return price_info

        except Exception as e:
            print(f"Error downloading group: {e}")
            return {}

    def _download_and_append(self, symbols, start_date, end_date, current_end_date, use_threads=True):
        """Download incremental data and append to existing files"""
        try:
            # Batch download incremental data
            new_data = yf.download(symbols, start=start_date, end=end_date,
                                  group_by='ticker', threads=use_threads, progress=False)
            price_info = {}
            # Append to each symbol's file
            for symbol in symbols:
                try:
                    if len(symbols) == 1:
                        incremental_df = new_data
                    else:
                        incremental_df = new_data[symbol]

                    if incremental_df.empty:
                        continue

                    file_path = self.data_dir / f"{symbol}.parquet"

                    # Load existing data
                    existing_df = pd.read_parquet(file_path)

                    # Combine and deduplicate
                    combined_df = pd.concat([existing_df, incremental_df])
                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                    combined_df.sort_index(inplace=True)

                    # Save updated data
                    combined_df.to_parquet(file_path)

                    # Update log
                    self._update_log_entry(symbol,
                                         existing_df.index.min().strftime('%Y-%m-%d'),
                                         end_date)
                    price_info[symbol] = combined_df
                except Exception as e:
                    print(f"    Error updating {symbol}: {e}")

            self._save_log()
            return price_info
        except Exception as e:
            print(f"  Error downloading incremental data: {e}")
            return {}

    def _update_log_entry(self, symbol, start_date, end_date):
        """Update or add log entry for a symbol"""
        mask = self.log_df['symbol'] == symbol
        if mask.any():
            self.log_df.loc[mask, 'end_date'] = end_date
        else:
            new_row = pd.DataFrame({'symbol': [symbol], 'end_date': [end_date]})
            self.log_df = pd.concat([self.log_df, new_row], ignore_index=True)

    def get_data(self, symbol):
        """Retrieve data for a symbol"""
        file_path = self.data_dir / f"{symbol}.parquet"
        if file_path.exists():
            return pd.read_parquet(file_path)
        return None


# ============== USAGE EXAMPLE ==============

if __name__ == "__main__":
    # Initialize updater
    updater = GroupedStockUpdater(
        data_dir='stock_data',
        log_file='stock_log.csv'
    )

    # Example: Update a batch of 600 symbols
    symbols_batch = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',  # Some existing
        'NVDA', 'META', 'AMD', 'NFLX', 'INTC',     # Mix of new and existing
        # ... add your 600 symbols here
    ]

    # Update to yesterday (or specify a date)
    updater.update_batch(symbols_batch)  # Uses yesterday by default
    # Or: updater.update_batch(symbols_batch, target_end_date='2025-10-02')

    # Get data for a specific symbol
    aapl_data = updater.get_data('AAPL')
    if aapl_data is not None:
        print(f"\nAAPL data shape: {aapl_data.shape}")
        print(aapl_data.tail())
