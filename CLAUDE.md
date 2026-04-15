# CLAUDE.md — CSP ML Pipeline

Cash-Secured Put (CSP) option trading ML pipeline.  
Rewritten from `csp_feature_lab2/` for cleaner structure and maintainability.

---

## Quick Start

```bash
pip install -r requirements.txt
```

Change 2 lines at the top of `config.yaml`, then run the pipeline in order:

```bash
python pipeline/a01_build_features.py   # build enriched dataset
python pipeline/a02_collect_events.py   # collect corporate events
python pipeline/a03_filter_trades.py    # filter trades near events
python pipeline/a04_label_data.py       # label with win/loss outcomes
python pipeline/a05_merge_datasets.py   # build rolling-window training set
python pipeline/b01_train_winner.py     # train winner classifier
python pipeline/b02_score_winner.py     # score new candidates
pytest test/                            # run tests
```

---

## Configuration

### The 2 active variables (top of config.yaml)

```yaml
active_score_dataset:   "f"   # short tag for the most-recently labeled batch
active_process_dataset: "f"   # set to the same tag while running a01–a04
rolling_window_weeks:   14    # how far back to look for training data
```

`active_train_profile` is gone.  The training window is computed automatically
by `a05_merge_datasets.py` from `rolling_window_weeks` and the batch registry.

### Adding a new batch — 6-step workflow

```
1. Add a new entry in common_configs (config.yaml) using the template at the bottom.
2. Set active_process_dataset to the new tag.
3. Run: a01 → a02 → a03 → a04   (processes + labels the new batch)
4. Set active_score_dataset to the same tag.
5. Run: a05                       (auto-selects the rolling training window)
6. Run: b01 → b02                 (train + score)
```

That's it — no cumulative tags, no manual file naming.

---

## Rolling Window Design

### Why rolling instead of expanding

With 11+ months of history (26+ biweekly batches), an expanding window
over-weights old market regimes (volatility environment, rates).  A fixed
14-week window keeps the model anchored to recent conditions.

### Window selection logic (`service/env_config.py :: get_rolling_train_batches`)

```
window_end   = events_start_date of active_score_dataset
window_start = window_end − rolling_window_weeks

included = batches where events_start_date ∈ [window_start, window_end)
           sorted chronologically; active_score_dataset excluded
```

Changing `rolling_window_weeks` in config.yaml is the only knob needed.
Try 16 weeks (~8 batches) if the model shows high variance.

### Output file naming convention

All output filenames are **date-stamped**, not tag-stamped.  The date used
is the `events_start_date` of `active_score_dataset` formatted as `YYYYMMDD`.

| File | Example (score dataset start = 2025-10-27) |
|------|--------------------------------------------|
| Merged training CSV | `output/data_merged/merged_roll14w_20251027.csv` |
| Model directory | `output/winner_train/v9_roll14w_20251027/` |
| Model file | `winner_classifier_model_20251027_lgbm.pkl` |
| Score output | `output/winner_score/v9_roll14w_20251027/scores_20251027.csv` |

The letter tags (`orig`, `a`, `b`, …) are **internal lookup keys only** — they
appear in `config.yaml → common_configs` and nowhere in any output filename.
They exist solely to let you reference a batch by a short name in the config.

### Template placeholders (resolved by `service/env_config.py`)

| Placeholder | Resolves to |
|-------------|-------------|
| `{active_score_dataset}` | tag string, e.g. `"f"` |
| `{active_process_dataset}` | tag string, e.g. `"f"` |
| `{active_score_date}` | `events_start_date` as YYYYMMDD, e.g. `"20251027"` |
| `{active_score_labeled_csv}` | `output_csv` of the score dataset config entry |
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
│   ├── a05_merge_datasets.py      step 5 (optional): walk-forward dataset merging
│   ├── b01_train_winner.py        train winner classifier with OOF cross-validation
│   └── b02_score_winner.py        score new candidates
│
├── service/                       ← reusable library modules
│   ├── constants.py               ← feature lists (BASE_FEATS, GEX_FEATS, NEW_FEATS)
│   │                                 and all magic numbers / defaults
│   ├── env_config.py              ← YAML + .env config loader with template resolution
│   ├── data_prepare.py            ← price caching, capital calc, macro feature engineering
│   ├── preprocess.py              ← DTE calc, normalised returns, GEX merge
│   ├── winner_scoring.py          ← threshold selection, model loading, prediction
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
output/data_prep/trades_with_gex_macro_<tag>.csv   + symbols file
  ↓  [a02_collect_events]
output/data_prep/corp_events/events_<tag>.csv
  ↓  [a03_filter_trades]
option/put/filtered/trades_filtered_<tag>.csv
  ↓  [a04_label_data]
output/data_labeled/labeled_<tag>_filtered.csv
  ↓  [a05_merge_datasets — optional]
output/data_merged/merged_with_gex_macro_<combo>.csv
  ↓  [a04_label_data — merge mode]
output/data_labeled/labeled_merged_with_gex_macro_<combo>.csv
  ↓  [b01_train_winner]
output/winner_train/v9_oof_<combo>/winner_classifier_model_<combo>_lgbm.pkl
  ↓  [b02_score_winner]
output/winner_score/v9_model_<combo>/scores_winner_lgbm_<tag>.csv
```

---

## Key Design Decisions

### Feature definitions in one place
All feature groups (`BASE_FEATS`, `GEX_FEATS`, `NEW_FEATS`) live in
`service/constants.py`.  Import from there; never redefine inline.

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

---

## Service Modules

| Module | Responsibility |
|--------|----------------|
| `constants.py` | Feature lists, magic numbers, pipeline defaults |
| `env_config.py` | YAML + .env config with template resolution |
| `data_prepare.py` | Price caching, capital calculation, macro features |
| `preprocess.py` | DTE calc, normalised returns, GEX snapshot merging |
| `winner_scoring.py` | Model loading, threshold selection, scoring |
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
