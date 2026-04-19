# CLAUDE.md — CSP ML Pipeline

Cash-Secured Put (CSP) option trading ML pipeline.  
Rewritten from `csp_feature_lab2/` for cleaner structure and maintainability.

---

## Quick Start

```bash
pip install -r requirements.txt
```

All option data lives in a single folder (`option/put/`).  When you have new
data, update the three date fields in the `dataset:` block of `config.yaml`
(see section 2 below), then run the pipeline in order:

```bash
# --- Data ingestion (per new batch) ---
python pipeline/a01_build_features.py   # build enriched dataset
python pipeline/a02_collect_events.py   # collect corporate events
python pipeline/a03_filter_trades.py    # filter trades near events
python pipeline/a04_label_data.py       # label with win/loss outcomes

# --- Training dataset ---
python pipeline/a05_merge_datasets.py   # build rolling-window training set

# --- Winner classifier ---
python pipeline/b01_train_winner.py     # train 4-bin winner classifier; produces winner_scores_oof.csv
python pipeline/b02_score_winner.py     # score new batch: win_proba reference signal

# --- Tail classifier  (MUST run AFTER b01 — depends on winner_scores_oof.csv) ---
python pipeline/b03_train_tail.py       # train tail veto classifier
python pipeline/b04_score_tail.py       # score: tail_proba reference signal

pytest test/                            # run tests
```

See `INSTRUCTIONS.md` for the full end-to-end workflow with decision guidance.

---

## Configuration

### The only knob in config.yaml

```yaml
rolling_window_weeks: 14   # how many weeks of labeled history to train on
```

When you have new option data, update three date fields in the `dataset:` block:

```yaml
dataset:
  data_dir:           "option/put"           # all snapshot CSVs here, no sub-folders
  data_basic_csv:     "trades_raw_current.csv"
  output_csv:         "labeled_trades_current.csv"
  cutoff_date:        "YYYY-MM-DD"           # ← last expiry in data + buffer
  events_start_date:  "YYYY-MM-DD"           # ← first trade date of scoring window
  events_end_date:    "YYYY-MM-DD"           # ← last trade date of scoring window
  ...
```

Output paths are automatically dated from `events_start_date` (YYYYMMDD suffix).
Example: with `events_start_date: 2025-10-27`, every output uses `20251027`.

### Updating to new data — 4-step workflow

```
1. Drop new snapshot CSVs into option/put/ (no sub-folders needed).
2. Update the three date fields in config.yaml → dataset.
3. Run: a01 → a02 → a03 → a04   (builds enriched + labeled dataset)
4. Run: a05                       (date-filters rolling training window from labeled CSV)
5. Run: b01 → b02                 (train 4-bin winner + score)
   Optional: b03 → b04            (train tail veto + score)
```

The rolling window is applied by `a05_merge_datasets.py` by filtering the
single labeled CSV on `tradeTime` rather than merging separate batch files.

---

## Rolling Window Design

### Why rolling instead of expanding

A fixed 14-week window keeps the model anchored to recent market conditions
rather than over-weighting older volatility regimes.

### Window selection logic (`a05_merge_datasets.py`)

```
window_end   = dataset.events_start_date   (first day of the scoring period)
window_start = window_end − rolling_window_weeks

training rows = labeled CSV rows where:
    window_start ≤ tradeTime < window_end
```

All option data is in one labeled CSV; `a05` date-filters it rather than
merging separate batch files.  Changing `rolling_window_weeks` is the only
knob.  Try 16 weeks if the model shows high variance.

### Output file naming convention

All output filenames are **date-stamped** from `dataset.events_start_date` (YYYYMMDD).

| File | Example (events_start_date = 2025-10-27) |
|------|------------------------------------------|
| Merged training CSV | `output/data_merged/merged_roll14w_20251027.csv` |
| Model directory | `output/winner_train/v9_roll14w_20251027/` |
| Model file | `winner_classifier_model_20251027_lgbm.pkl` |
| Score output | `output/winner_score/v9_roll14w_20251027/scores_20251027.csv` |

### Template placeholders (resolved by `service/env_config.py`)

| Placeholder | Resolves to |
|-------------|-------------|
| `{active_score_date}` | `dataset.events_start_date` as YYYYMMDD, e.g. `"20251027"` |
| `{active_score_labeled_csv}` | `dataset.output_csv`, e.g. `"labeled_trades_current.csv"` |
| `{rolling_window_weeks}` | integer window size, e.g. `"14"` |
| `{model_type}` | `winner.model_type`, e.g. `"lgbm"` |

---

## Project Structure

```
csp_ml_rewrite/
├── config.yaml                    ← main configuration (edit the 2 active_* lines + rolling_window_weeks)
├── corp_action_config.yaml        ← corporate-event filtering windows
├── requirements.txt
│
├── pipeline/                      ← executable pipeline scripts (run in order)
│   ├── a01_build_features.py      step 1: load snapshots, merge GEX, add macro features
│   ├── a02_collect_events.py      step 2: scrape EDGAR earnings + yfinance splits
│   ├── a03_filter_trades.py       step 3: drop trades near earnings / splits
│   ├── a04_label_data.py          step 4: fetch expiry prices, compute PnL, label win/loss
│   ├── a05_merge_datasets.py      step 5: build rolling-window training set
│   ├── b01_train_winner.py        train winner classifier (likely profitable trades)
│   ├── b02_score_winner.py        score new candidates with winner model
│   ├── b03_train_tail.py          train tail classifier (likely catastrophic losers)
│   └── b04_score_tail.py          flag likely fat-tail losers
│
├── service/                       ← reusable library modules
│   ├── constants.py               ← feature lists (BASE_FEATS, GEX_FEATS, NEW_FEATS,
│   │                                 TAIL_FEATS, TAIL_EARNINGS_FEATS) + defaults
│   ├── env_config.py              ← YAML + .env config loader with template resolution
│   ├── data_prepare.py            ← price caching, capital calc, macro feature engineering
│   ├── preprocess.py              ← DTE calc, normalised returns, GEX merge
│   ├── winner_scoring.py          ← winner model loading, threshold selection, scoring
│   ├── tail_scoring.py            ← tail model loading, threshold selection, scoring
│   ├── utils.py                   ← shared prep functions, threshold helpers
│   ├── stock_data_manager.py      ← batch price updates with caching
│   ├── production_data.py         ← feature engineering for live / on-fly scoring
│   ├── split_detector.py          ← stock split detection via yfinance
│   ├── nasdaq_earnings.py         ← Nasdaq earnings calendar scraper
│   ├── get_vix.py                 ← real-time VIX fetch (Selenium)
│   └── option_metrics.py          ← option-specific calculations
│
├── scripts/                       ← utility / maintenance scripts
│   ├── build_all_datasets.py      process all common_config datasets in one run
│   └── daily_stock_update.py      refresh price cache for active symbols
│
├── eval/                          ← evaluation and diagnostics
│   └── eval_classifier.py         ROC-AUC, PR-AUC, threshold sweep for any model
│
├── data/                          ← small data files tracked in git
│   ├── missing_stocks.json        symbols that have no price data (auto-managed)
│   └── exclude_stocks.json        manually curated exclusion list
│
└── test/                          ← pytest test suite
```

---

## Data Flow

```
Raw CSP snapshots (option/put/put25_*/coveredPut_*.csv)
  ↓  [a01_build_features]
output/data_prep/trades_with_gex_macro_<section>.csv   + symbols file
  ↓  [a02_collect_events]
output/data_prep/corp_events/events_<tag>.csv
  ↓  [a03_filter_trades]
option/put/filtered/trades_filtered_<tag>.csv
  ↓  [a04_label_data]
output/data_labeled/labeled_trades_<section>.csv
  ↓  [a05_merge_datasets]
output/data_merged/merged_roll{W}w_{YYYYMMDD}.csv     ← rolling window of last W weeks
          │
          ↓  [b01_train_winner]
output/winner_train/v9_roll{W}w_{YYYYMMDD}/
  winner_classifier_model_{YYYYMMDD}_lgbm.pkl          ← 4-bin multiclass model
  winner_scores_oof.csv                                ← OOF probabilities (p_bin0-3)
          │
          ├──────────────────────────────────────────┐
          ↓  [b02_score_winner]                      ↓  [b03_train_tail]  ← REQUIRES b01 OOF
output/winner_score/v9_roll{W}w_{YYYYMMDD}/          output/tails_train/v9_roll{W}w_{YYYYMMDD}/
  scores_{YYYYMMDD}.csv                                tail_classifier_model_{YYYYMMDD}.pkl
                                                       ↓  [b04_score_tail] ← applies winner first
                                                     output/tails_score/v9_roll{W}w_{YYYYMMDD}/
                                                       tail_scores_{YYYYMMDD}.csv
```

### Three-layer defense system

| Layer | Script | Output |
|-------|--------|--------|
| 1. Day gate | (upstream) | only trade on eligible days |
| 2. Winner ranker | b01 / b02 | `win_proba` — 4-bin ranking; prefer bin-3 trades |
| 3. Tail veto | b03 / b04 | `tail_proba` — flag catastrophic-loss risk |

**b03 depends on b01.**  The tail classifier uses the winner model's per-class OOF
probabilities (`p_bin0-3`, `conflict_score = p_bin0 × p_bin3`) as features.
Always run b01 before b03.  b04 likewise loads the winner model to compute
those features for new trades before applying the tail model.

See `INSTRUCTIONS.md` for full workflow guidance.

---

## Key Design Decisions

### Feature definitions in one place
All feature groups (`BASE_FEATS`, `GEX_FEATS`, `NEW_FEATS`, `TAIL_FEATS`,
`BIN_PROB_FEATS`) live in `service/constants.py`.  Import from there; never
redefine inline.

### Dynamic dataset tag resolution
`service/env_config.py::get_active_dataset_config()` derives the dataset tag
from each config's `data_basic_csv` field dynamically — no hard-coded
`tag_to_key` mapping that needs manual updates.

### Rolling window training (not expanding)
`service/env_config.py::get_rolling_train_batches()` selects the last
`rolling_window_weeks` weeks of labeled batches automatically.  No cumulative
tags like `origabcde` — just set `rolling_window_weeks` and `active_score_dataset`.

### Date-stamped outputs (not tag-stamped)
All merged datasets, model directories, and score files use the
`events_start_date` of the active score batch (YYYYMMDD) in their names.
The short letter tags (`a`, `b`, …) never appear in output paths.

### Symbol exclusions from data files (not code)
`pipeline/a04_label_data.py` loads exclusions from:
- `data/missing_stocks.json` — symbols without price data
- `data/exclude_stocks.json` — manually curated

Add problematic symbols to these JSON files instead of editing Python code.

### Config template resolution
Paths like `"output/winner_train/v9_roll{rolling_window_weeks}w_{active_score_date}/"` are
resolved by `service/env_config.py` whenever `getenv()` is called.
See "Rolling Window Design" section above for the full placeholder table.

### bins4 multiclass — critical correctness constraints

These were identified as bugs and fixed (see `csp_feature_lab2/doc/bins4_quick_fixes.md`):

**1. Label/target alignment (a04_label_data.py)**
`y_bin` and the per-day quartile assignment must use the **same column** as
`WINNER_TRAIN_TARGET` (default: `return_mon`), not `return_pct`.
`a04_label_data.py::build_labeled_dataset()` now computes `return_per_day`,
`return_ann`, and `return_mon` before calling `_assign_bins`, and `_assign_bins`
groups by trade date and applies quartile cuts to `return_mon`.

**2. LightGBM eval_metric for multiclass (b01_train_winner.py)**
When `label_mode == "bins4"`, LightGBM uses `objective="multiclass"` which
requires `eval_metric="multi_logloss"`.  Using `"aucpr"` (the binary default)
causes a crash.  The metric is now selected conditionally:
```python
lgbm_metric = "multi_logloss" if self.cfg.label_mode == "bins4" else "aucpr"
```

**3. OOF NaN contamination in metrics (b01_train_winner.py)**
Early rows in time-series CV never appear in the validation fold (`fold_idx == -1`).
Their OOF probability rows remain `np.nan`.  `np.argmax` on NaN rows silently
returns 0, corrupting accuracy and F1.  `save_model_pack` now receives `fold_idx`
and filters with `valid = (fold_idx != -1)` before computing any metrics.

**4. Top-vs-bottom spread metric (b01_train_winner.py)**
The primary trading-quality signal — mean return of top-10% trades by `p_bin3`
minus mean return of bottom-10% — was missing from model metrics.  It is now
computed in `save_model_pack` and stored as `top_minus_bottom_spread` in the
metrics JSON alongside `top10pct_mean_target`, `bottom10pct_mean_target`, and
`n_oof_valid`.

---

## Service Modules

| Module | Responsibility |
|--------|----------------|
| `constants.py` | Feature lists, magic numbers, pipeline defaults |
| `env_config.py` | YAML + .env config with template resolution |
| `data_prepare.py` | Price caching, capital calculation, macro features |
| `preprocess.py` | DTE calc, normalised returns, GEX snapshot merging |
| `winner_scoring.py` | Winner model loading, threshold selection, scoring |
| `tail_scoring.py` | Tail model loading, TailModelPack, threshold selection, scoring |
| `utils.py` | Data-prep helpers, threshold-search utilities |
| `stock_data_manager.py` | Batch yfinance price downloads with parquet cache |
| `production_data.py` | On-fly feature engineering for live data |
| `split_detector.py` | Stock split detection |
| `nasdaq_earnings.py` | Earnings calendar from Nasdaq |
| `get_vix.py` | Real-time VIX scraping |
| `option_metrics.py` | Option-specific metric helpers |

---

## Model Types

Set `winner.model_type` in `config.yaml`:

| Value | Description |
|-------|-------------|
| `lgbm` | LightGBM (default, fastest) |
| `catboost` | CatBoost (good out-of-the-box) |
| `rf` | RandomForest (interpretable) |

---

## Output Directory Structure

```
output/
├── data_prep/           enriched feature datasets (pre-labeling)
├── data_labeled/        labeled datasets with win/loss + returns
├── data_merged/         walk-forward merged training sets
├── winner_train/        trained winner models + OOF metrics
├── winner_score/        scored candidate files
├── tails_train/         tail-loss models (experimental)
├── tails_score/         tail-loss predictions
├── eval/                evaluation reports
├── price_cache/         yfinance parquet cache (one file per symbol)
└── vix_data.csv         VIX historical data
```

---

## Important Notes

- **GEX data**: symlinked from NAS at `gex101 → /mnt/nas_share/dev/data/gex101/processed/csv`
- **Cutoff dates**: each dataset config has a `cutoff_date` — labeling only processes
  trades with `expirationDate ≤ cutoff_date` to prevent future-price leakage
- **DTE filter**: trades with `daysToExpiration > 14` are excluded throughout
- **Sample weighting**: optional; emphasises trades with larger |return|
- **Threshold calibration**: auto-calibrates on validation set or uses best-F1 from training

---

## Testing

```bash
pytest test/ -v                         # all tests
pytest test/test_preprocess.py -v       # single test module
```

---

## Dependencies

Key packages: `pandas numpy scipy scikit-learn lightgbm catboost yfinance pyarrow
fastparquet exchange_calendars python-dotenv pyyaml requests matplotlib joblib`

Full list in `requirements.txt`.
