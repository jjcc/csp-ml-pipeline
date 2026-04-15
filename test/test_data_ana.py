import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import joblib
import pandas as pd
from service.preprocess import add_dte_and_normalized_returns
from service.utils import prep_tail_training_df, fill_features_with_training_medians
from service.utils import ALL_FEATS
from train_tail_with_gex import _fill_features

class TestDataAna(unittest.TestCase):

    def setUp(self):
        self.df_enriched = pd.read_csv("output/labeled_trades_enriched.csv")

    
    def test_compare_potentialann_realmon(self):
        """
        compare the columns of basic, gex, and macro feature datasets
        """
        df_enriched = self.df_enriched
        # filter the potentialReturnAnnual < 400
        df_enriched = df_enriched[df_enriched['potentialReturnAnnual'] < 100]
        # lable the return_pct > 0 as 1, else 0
        df_enriched['labelled_winner'] = (df_enriched['return_pct'] > 0).astype(int)



        # add the monthly return column
        df_enriched = add_dte_and_normalized_returns(df_enriched)
        c1 = df_enriched['potentialReturnAnnual']
        c2 = df_enriched['return_mon'] * 12

        # plot the two columns against each other
        import matplotlib.pyplot as plt
        # smaller dot, bitter chart image
        plt.figure(figsize=(8, 6))
        plt.scatter(c1, c2, alpha=0.5, s=5)
        plt.xlabel('potentialReturnAnnual')
        plt.ylabel('return_mon * 12')
        plt.title('Comparison of potentialReturnAnual and return_mon * 12')
        plt.grid(True)
        #plt.plot([c1.min(), c1.max()], [c1.min(), c1.max()], 'r--')  # y=x line
        plt.savefig('test/data/potentialann_vs_realmon_100max_b.png')



        #gex extra = {'gex_pos', 'gex_missing', 'gex_distance_to_flip', 'gex_total_abs', 'symbol_norm', 'gex_sign_at_ul', 'gex_flip_strike', 'gex_file', 'trade_date', 'gex_neg', 'gex_center_abs_strike', 'gex_total', 'gex_gamma_at_ul'}
        #macro extra = {'VIX', 'ret_5d_norm', 'prev_close_minus_strike', 'ret_2d', 'prev_close_minus_strike_pct', 'log1p_DTE', 'prev_close', 'ret_5d', 'ret_2d_norm'}

    def test_win_rate_by_potential_return_quantiles(self):
        """
        Analyze win rate across different quantiles of potentialReturnAnnal
        """
        df_enriched = self.df_enriched.copy()
        
        # Add labelled_winner column if not exists
        if 'labelled_winner' not in df_enriched.columns:
            df_enriched['labelled_winner'] = (df_enriched['return_pct'] > 0).astype(int)
        
        # Define quantile thresholds
        quantiles = [30, 40, 50, 60, 70, 90, 120, 150, 200]
        
        # Calculate quantile values
        quantile_values = []
        for q in quantiles:
            if q <= 100:
                quantile_val = df_enriched['potentialReturnAnnual'].quantile(q/100)
            else:
                # For values > 100, interpret as absolute thresholds
                quantile_val = q
            quantile_values.append(quantile_val)
        
        # Create segments
        segments = []
        prev_val = df_enriched['potentialReturnAnnual'].min()
        
        for i, (q, val) in enumerate(zip(quantiles, quantile_values)):
            if q <= 100:
                # Quantile-based segment
                mask = (df_enriched['potentialReturnAnnual'] > prev_val) & (df_enriched['potentialReturnAnnual'] <= val)
                segment_name = f"Q{q} ({prev_val:.1f} to {val:.1f})"
            else:
                # Absolute threshold segment
                if i == 0 or quantiles[i-1] <= 100:
                    # First absolute threshold or transition from quantile
                    if i > 0:
                        prev_val = quantile_values[i-1]
                mask = (df_enriched['potentialReturnAnnual'] > prev_val) & (df_enriched['potentialReturnAnnual'] <= val)
                segment_name = f"{prev_val:.1f} to {val:.1f}"
            
            segment_data = df_enriched[mask]
            
            if len(segment_data) > 0:
                win_rate = segment_data['labelled_winner'].mean() * 100
                count = len(segment_data)
                wins = segment_data['labelled_winner'].sum()
                
                segments.append({
                    'Segment': segment_name,
                    'Min': segment_data['potentialReturnAnnual'].min(),
                    'Max': segment_data['potentialReturnAnnual'].max(),
                    'Count': count,
                    'Wins': wins,
                    'Win Rate (%)': win_rate
                })
            
            prev_val = val
        
        # Add final segment for values above the last threshold
        mask = df_enriched['potentialReturnAnnual'] > quantile_values[-1]
        segment_data = df_enriched[mask]
        
        if len(segment_data) > 0:
            win_rate = segment_data['labelled_winner'].mean() * 100
            count = len(segment_data)
            wins = segment_data['labelled_winner'].sum()
            
            segments.append({
                'Segment': f"> {quantile_values[-1]:.1f}",
                'Min': segment_data['potentialReturnAnnual'].min(),
                'Max': segment_data['potentialReturnAnnual'].max(),
                'Count': count,
                'Wins': wins,
                'Win Rate (%)': win_rate
            })
        
        # Create DataFrame and display results
        results_df = pd.DataFrame(segments)
        
        print("\n" + "="*80)
        print("Win Rate Analysis by potentialReturnAnnual Segments")
        print("="*80)
        print(results_df.to_string(index=False))
        
        # Summary statistics
        print("\n" + "-"*80)
        print("Summary Statistics:")
        print(f"Total samples: {df_enriched.shape[0]}")
        print(f"Overall win rate: {df_enriched['labelled_winner'].mean()*100:.2f}%")
        print(f"potentialReturnAnnual range: [{df_enriched['potentialReturnAnnual'].min():.2f}, {df_enriched['potentialReturnAnnual'].max():.2f}]")
        
        # Save results to CSV
        results_df.to_csv('test/data/win_rate_by_potential_return_segments.csv', index=False)
        print("\nResults saved to: test/data/win_rate_by_potential_return_segments.csv")
        
        return results_df


    def test_win_rate_by_prodictions(self):
        """
        Analyze win rate across different quantiles of model predictions
        """
        data_file = "output/winner_score/scores_winner_model_v6_oof_ne_straited.csv"
        df_raw = pd.read_csv(data_file)
        df_data = df_raw.copy()

        conditions = {"no_less_than_59pct_potentialReturnAnnual": df_data['potentialReturnAnnual'] >= 59,
                      "no_more_than_200pct_potentialReturnAnnual": df_data['potentialReturnAnnual'] <= 200,
                      "no_less_than_59pct_and_no_more_than_200pct_potentialReturnAnnual": (df_data['potentialReturnAnnual'] >= 59) & (df_data['potentialReturnAnnual'] <= 200),
                      "all_data": pd.Series([True]*len(df_data))}
        for cond_name, cond in conditions.items():
            print(f"Filtering data: {cond_name} ({cond.sum()} samples)")
            df_data = df_raw.copy()
            df_data = df_data[cond]

            # Add labelled_winner column if not exists
            results_df = self.calc_quantile(df_data, cond_name=cond_name)
            # Save results to CSV
            results_df.to_csv(f'test/data/win_rate_ana_{cond_name}.csv', index=False)

    def calc_quantile(self, df_data, cond_name=""):
        if 'labelled_winner' not in df_data.columns:
            df_data['labelled_winner'] = df_data['win_labeled'] > 0
        
        
        
        #df_data['win_proba'] = clf.predict_proba(X)[:, 1]
        
        # Define quantile thresholds
        #quantiles = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        #quantiles = [40, 50, 60, 70, 80, 90, 95]
        quantiles = [2, 5, 10, 20, 30, 40, 50, 60, 80]
        
        # Calculate quantile values
        quantile_values = [df_data['win_proba'].quantile(q/100) for q in quantiles]
        
        # Create segments
        segments = []
        prev_val = df_data['win_proba'].min()
        
        for q, val in zip(quantiles, quantile_values):
            mask = (df_data['win_proba'] > prev_val) & (df_data['win_proba'] <= val)
            segment_data = df_data[mask]
            
            if len(segment_data) > 0:
                win_rate = segment_data['labelled_winner'].mean() * 100
                count = len(segment_data)
                wins = segment_data['labelled_winner'].sum()
                
                segments.append({
                    'Segment': f"Q{q} ({prev_val:.3f} to {val:.3f})",
                    'Min': segment_data['win_proba'].min(),
                    'Max': segment_data['win_proba'].max(),
                    'Count': count,
                    'Wins': wins,
                    'Win Rate (%)': win_rate
                })
            
            prev_val = val
        
        # Add final segment for values above the last threshold
        mask = df_data['win_proba'] > quantile_values[-1]
        segment_data = df_data[mask]
        if len(segment_data) > 0:
            win_rate = segment_data['labelled_winner'].mean() * 100
            count = len(segment_data)
            wins = segment_data['labelled_winner'].sum()
            
            segments.append({
                'Segment': f"> {quantile_values[-1]:.3f}",
                'Min': segment_data['win_proba'].min(),
                'Max': segment_data['win_proba'].max(),
                'Count': count,
                'Wins': wins,
                'Win Rate (%)': win_rate
            })
        # Create DataFrame and display results
        results_df = pd.DataFrame(segments)
        print("\n" + "="*80)
        print("Win Rate Analysis by Model Prediction Segments: " + cond_name)
        print("="*80)
        print(results_df.to_string(index=False))
        # Summary statistics
        print("\n" + "-"*80)
        print("Summary Statistics:")
        print(f"Total samples: {df_data.shape[0]}")
        print(f"Overall win rate: {df_data['labelled_winner'].mean()* 100:.2f}%")
        print(f"Model prediction range: [{df_data['win_proba'].min():.3f}, {df_data['win_proba'].max():.3f}]")
        return results_df

    def test_unique_symbols(self):
        """
        Test for unique symbols in one day's data.
        """
        folder = "prod/output"
        date = "2025-09-18"
        #files = [f for f in os.listdir(folder) if f.startswith(f"scored_tail_winner_lgbm_{date}_") and f.endswith(".csv")]
        files = [f for f in os.listdir(folder) if f.startswith(f"scored_tail_winner_test_{date}_") and f.endswith(".csv")]
        files.sort()
        unique_basesymbols = []
        unique_symbols=set()
        common_symbols = set()
        prev_symbols = set()
        print("###################################")
        for file in files:
            if '_00.' in file:
                continue
            path = os.path.join(folder, file)
            df = pd.read_csv(path)
            unique_basesymbols.extend(df['baseSymbol'].unique())
            unique_symbols.update(df['symbol'].unique())
            if len(prev_symbols) == 0:
                prev_symbols = df['symbol'].unique()
                continue
            common_symbols = set(prev_symbols).intersection(set(df['symbol'].unique()))
            prev_symbols = df['symbol'].unique()
            # overlap with previous file
            print(f"File: {file}, unique symbols: {len(df['symbol'].unique())}, common with previous: {len(common_symbols)}")

        # unique base symbols, i.e. underlying stocks
        unique_basesymbols = set(unique_basesymbols)
        # unique symbols,i.e. option contracts
        unique_symbols = set(unique_symbols)
        print(f"unique symbols in raw data files: {len(unique_symbols)}")
        print(f"Unique base symbols in data files: {len(unique_basesymbols)}")
        self.assertTrue(len(unique_basesymbols) > 0, "There should be at least one unique symbol.")

if __name__ == '__main__':
    unittest.main()
