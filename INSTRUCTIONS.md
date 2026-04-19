# CSP ML Pipeline — Operating Instructions

This pipeline trains and applies a two-stage scoring system on Cash-Secured Put (CSP) option trades:

- **Winner Classifier (b01/b02)** — 4-bin multiclass model that ranks trades into quartiles (bin-0=worst, bin-3=best).  Produces `win_proba` and `p_bin3` reference scores.
- **Tail Classifier (b03/b04)** — veto-layer binary classifier that identifies trades likely to produce catastrophic losses (bin-0 outcomes).  Produces `tail_proba`.  **Depends on b01** — it uses the winner model's 4-bin probabilities as input features.

Both classifiers produce **reference scores**, not final decisions. You decide how to interpret and combine them based on your own risk tolerance and market view.

You can run only the winner classifier (b01/b02) independently.  To run the tail classifier (b03/b04), you must first run b01 for the same training window.

---

## 1. First-Time Setup

```bash
cd csp_ml_rewrite
pip install -r requirements.txt
```

Copy the environment template and set your paths:

```bash
cp .env.sample .env
# Edit .env if you need to override any config.yaml defaults
```

Verify the GEX symlink exists (or update `gex.base_dir` in config.yaml):

```bash
ls gex101/       # should show dated CSV folders
```

---

## 2. Understanding `config.yaml`

Only three lines at the top of `config.yaml` need to change as you work:

```yaml
active_score_dataset:   "f"   # tag of the batch you just finished labeling
active_process_dataset: "f"   # set this while running a01–a04 for a new batch
rolling_window_weeks:   14    # training window (14 weeks ≈ 7 biweekly batches)
```

**All output paths are automatically dated** from the `events_start_date` of
`active_score_dataset`.  For example, with `active_score_dataset: "f"` and
`events_start_date: 2025-10-27`, every output uses the suffix `20251027`.

You never need to type paths, dates, or cumulative tag strings like `origabcde`.

---

## 3. Adding a New Batch (Full Workflow)

Do this every time you have a new 2-week block of option data ready.

### Step 1 — Register the batch in `config.yaml`

Add an entry to `common_configs` following the template at the bottom of the
`common_configs` section.  Fill in:

| Field | Example |
|-------|---------|
| `data_dir` | `"option/put/put25_1110-1121"` |
| `data_basic_csv` | `"trades_raw_g_1110.csv"` |
| `output_csv` | `"labeled_trades_g_1110.csv"` |
| `cutoff_date` | `"2025-11-29"` (expiry date of last contract + buffer) |
| `events_start_date` | `"2025-11-10"` |
| `events_end_date` | `"2025-11-21"` |
| `events_output` | `"output/data_prep/corp_events/events_g.csv"` |
| `tickers_file` | `"output/data_prep/corp_events/symbols_in_option_data_g.txt"` |
| `filtered_trades_csv` | `"option/put/filtered/trades_filtered_g.csv"` |
| `filtered_out_csv` | `"option/put/filtered/trades_excluded_g.csv"` |

The tag (`g` here) is just an internal key.  It never appears in output filenames.

### Step 2 — Set active_process_dataset

```yaml
active_process_dataset: "g"
```

### Step 3 — Run data ingestion (a01 → a04)

```bash
python pipeline/a01_build_features.py   # load snapshots, merge GEX, add macro features
python pipeline/a02_collect_events.py   # scrape EDGAR earnings + yfinance splits
python pipeline/a03_filter_trades.py    # drop trades within event windows
python pipeline/a04_label_data.py       # fetch expiry prices, compute PnL, label win/loss
```

Each step reads `active_process_dataset` from config and operates only on batch "g".
Run them in order; each is a prerequisite for the next.

### Step 4 — Set active_score_dataset

```yaml
active_score_dataset: "g"
```

This is the batch you will score with the newly trained models.

### Step 5 — Build the rolling training set

```bash
python pipeline/a05_merge_datasets.py
```

This selects the last `rolling_window_weeks` weeks of completed batches
*before* batch "g", merges them, and writes:

```
output/data_merged/merged_roll14w_20251110.csv
```

The console output shows which batches were selected and how many rows.

### Step 6 — Train and score

#### Step 6a — Winner Classifier (required first)

```bash
python pipeline/b01_train_winner.py
```

Trains a 4-bin multiclass LightGBM that ranks trades into quartile bins
(bin-0 = worst return, bin-3 = best return) within each trading day.

Outputs:

```
output/winner_train/v9_roll14w_20251110/
  winner_classifier_model_20251110_lgbm.pkl
  winner_scores_oof.csv           ← also used as input to b03
  winner_classifier_metrics.json
  threshold_table.csv
  precision_recall_coverage.png
```

Key metrics to check: **bin-3 precision** and **bin-3 recall** printed at the end.

Then score the new batch:

```bash
python pipeline/b02_score_winner.py
```

Winner scores → `output/winner_score/v9_roll14w_20251110/scores_20251110.csv`

#### Step 6b — Tail Classifier (optional; run AFTER b01)

The tail classifier is a veto layer.  It uses the winner model's 4-bin OOF
probabilities (`p_bin0-3`, `conflict_score`) as features, so b01 must run first.

```bash
python pipeline/b03_train_tail.py
```

Outputs:

```
output/tails_train/v9_roll14w_20251110/
  tail_classifier_model_20251110.pkl
  tail_scores_oof.csv
  tail_classifier_metrics.json
  tail_feature_importances.csv
```

Key metric: **OOF AUC-PRC** (precision-recall AUC is more informative than
ROC-AUC for the imbalanced tail class).

Then score the new batch (b04 also applies the winner model first internally):

```bash
python pipeline/b04_score_tail.py
```

Tail scores → `output/tails_score/v9_roll14w_20251110/tail_scores_20251110.csv`

---

## 4. Reading the Score Files

### Winner score columns

| Column | Meaning |
|--------|---------|
| `p_bin3` | Probability trade falls in best quartile (0–1); use as the primary rank signal |
| `p_bin0` | Probability trade falls in worst quartile (0–1) |
| `win_proba` | Composite score (same as `p_bin3` in bins4 mode) |
| `win_predict` | 1 = predicted top-bin at chosen threshold |

### Tail score columns

| Column | Meaning |
|--------|---------|
| `tail_proba` | Probability the trade is a catastrophic loser (0–1) |
| `is_tail_pred` | 1 = predicted tail / avoid |

### Using scores together (optional)

If you want to look at both scores side by side, you can join the two output files on `symbol` + `tradeTime`:

```python
import pandas as pd

winner = pd.read_csv("output/winner_score/v9_roll14w_20251110/scores_20251110.csv")
tail   = pd.read_csv("output/tails_score/v9_roll14w_20251110/tail_scores_20251110.csv")

combined = winner.merge(
    tail[["symbol", "tradeTime", "tail_proba", "is_tail_pred"]],
    on=["symbol", "tradeTime"], how="left"
)
```

`combined` then has both `win_proba` and `tail_proba` as reference columns.
How you filter or rank the trades is entirely up to you.

---

## 5. Evaluating Model Quality

```bash
python eval/eval_classifier.py
```

By default this evaluates the winner model.  To evaluate the tail model,
update the `evaluation` section of `config.yaml`:

```yaml
evaluation:
  input:      "output/tails_score/v9_roll14w_20251110/tail_scores_20251110.csv"
  proba_col:  "tail_proba"
  label_mode: "tail_pct"    # labels worst 5% by return_mon
  tail_k:     0.05
```

---

## 6. Configuration Reference

### Rolling window size

| `rolling_window_weeks` | Batches (≈2 weeks each) | Approx. training rows |
|------------------------|-------------------------|----------------------|
| 12 | 6 | ~1,800–3,000 |
| 14 (default) | 7 | ~2,100–3,500 |
| 16 | 8 | ~2,400–4,000 |

Increase to 16 if the model shows high variance (OOF metrics unstable across runs).
Decrease if you suspect a recent market regime change dominates.

### Tail classifier parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `tail.winner_oof_csv` | auto | Path to winner_scores_oof.csv; resolved from active_score_date |
| `tail.with_earnings` | 1 | Include earnings-window features if present in data |
| `tail.model_name` | `tail_classifier_model` | Filename prefix for saved model pack |

Labeling is fixed: `tail = 1` when `y_true == 0` (actual bin-0 / worst quartile from winner).
Cross-validation reuses the winner fold structure (same fold column from winner_scores_oof.csv).
The tail model always uses LightGBM with isotonic calibration — no separate `model_type` knob.

`tailscoring.winner_model_in` must also point to the winner model .pkl so that b04
can compute `p_bin0-3` features for new (unlabeled) trades before tail scoring.

### Winner classifier parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `winner.model_type` | `lgbm` | Also `catboost`, `rf` |
| `winner.label_mode` | `bins4` | `bins4` = 4-bin multiclass (default); `binary` = win/loss |
| `winner.bins_mode` | `per_day` | `per_day` = quartiles within each trade date; `global` = absolute |
| `winner.bins_q` | `[0.25,0.5,0.75]` | Cut-points for 4 bins |
| `winner.bins_min_group` | 20 | Min trades per day to compute stable quartiles |
| `winner.time_col` | `captureTime` | Column used to group trades by day |
| `winner.target_precision` | `0.88,0.92` | Threshold sweep targets (binary mode only) |
| `winnerscore.auto_calibrate` | 1 | Auto-picks threshold on held-out data |

---

## 7. Maintenance

### Daily price cache update

```bash
python scripts/daily_stock_update.py
```

Keeps the parquet price cache current so VIX and macro features are accurate.

### Processing all historical batches at once

```bash
python scripts/build_all_datasets.py
```

Iterates every entry in `common_configs` and runs a01→a04 for each.
Use this when ingesting a large backlog (e.g., 11 months of data).

### Running tests

```bash
pytest test/ -v                         # all tests
pytest test/test_preprocess.py -v       # single module
```

---

## 8. Output Directory Map

```
output/
├── data_prep/           enriched feature datasets (after a01)
├── data_labeled/        win/loss labeled datasets (after a04)
├── data_merged/         rolling-window training sets (after a05)
├── winner_train/        winner models + OOF metrics (after b01)
├── winner_score/        winner scored batches (after b02)
├── tails_train/         tail models + OOF metrics (after b03)
├── tails_score/         tail scored batches (after b04)
├── eval/                evaluation reports (eval_classifier.py)
└── price_cache/         yfinance parquet cache (one file per symbol)
```

All subdirectories under `winner_train/`, `winner_score/`, `tails_train/`, and
`tails_score/` are named with the pattern `v9_roll{W}w_{YYYYMMDD}` where `W`
is the window size and `YYYYMMDD` is the score batch's start date — so you can
tell at a glance which model applies to which time period.
