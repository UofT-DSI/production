# stock_prices

Ingests raw CSV stock price data, partitions it by ticker and year into parquet, and computes a lagged-return feature dataset.

---

## `DataManager`

The single public class in this module. Supports two independent phases that can be run together or separately.

### Constructor

```python
DataManager(
    csv_dir: str | None = PRICE_CSV_DATA,
    price_dir: str | None = PRICE_DATA,
    features_path: str | None = FEATURES_DATA,
    tickers_file: str | None = TICKERS,
    n_sample: int = 30,
    random_state: int | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csv_dir` | `str \| None` | `PRICE_CSV_DATA` env var | Directory tree containing raw CSV files (one file per ticker). |
| `price_dir` | `str \| None` | `PRICE_DATA` env var | Root directory for partitioned parquet output. |
| `features_path` | `str \| None` | `FEATURES_DATA` env var | Output path for the featurized parquet dataset. |
| `tickers_file` | `str \| None` | `TICKERS` env var | Path to the tickers file (reserved for future use). |
| `n_sample` | `int` | `30` | Number of tickers sampled by `process_sample_files()`. |
| `random_state` | `int \| None` | `None` | Random seed for sampling; `None` gives non-deterministic results. |

### Environment variables

| Variable | Purpose |
|----------|---------|
| `PRICE_CSV_DATA` | Root directory containing raw CSV files. |
| `PRICE_DATA` | Root directory for partitioned parquet output. |
| `FEATURES_DATA` | Output path for the featurized parquet dataset. |
| `TICKERS` | Path to the tickers file (reserved for future use). |

---

## Ingest phase

Reads raw CSVs, partitions by ticker and year, and writes parquet datasets.

Output layout: `<price_dir>/<TICKER>/<TICKER>_<YEAR>/`

### Public methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `process_all_files` | `() -> None` | Ingest every CSV found under `csv_dir`. |
| `process_sample_files` | `(n_sample=None, random_state=None) -> None` | Ingest a random sample. Parameters override instance values for this call only. |
| `get_file_list` | `() -> None` | Populate `self.file_list` from all CSVs under `csv_dir`. |
| `select_sample` | `() -> None` | Down-sample `self.file_list` to `self.n_sample` entries in place. |
| `get_data_and_save_by_year` | `() -> None` | Iterate `self.file_list`, read each CSV, delegate to `save_by_year`. |
| `save_by_year` *(static)* | `(price_dt, out_dir) -> None` | Partition a DataFrame by ticker/year and write parquet. |
| `get_stock_price_data` *(static)* | `(stock_file) -> pd.DataFrame` | Read a CSV and attach `source` and `ticker` metadata columns. |

---

## Featurize phase

Loads the partitioned parquet output, computes a one-day lagged close and simple daily return, and writes a single feature dataset.

**Features added:**

| Column | Formula |
|--------|---------|
| `Close_lag_1` | `Close` shifted by 1 row within each ticker (sorted by `Date`). |
| `Returns` | `Close / Close_lag_1 - 1` |

### Public methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `featurize` | `() -> None` | Run `load_prices`, `create_features`, and `save_features` in sequence. |
| `load_prices` | `() -> None` | Read all parquet files under `price_dir` into `self.price_dd`. |
| `create_features` | `() -> None` | Compute lag and returns; store result in `self.features`. |
| `save_features` | `() -> None` | Write `self.features` to parquet at `self.features_path`. |

---

## End-to-end example

```python
import os
from stock_prices.data_manager import DataManager

dm = DataManager(
    csv_dir="data/raw/csv",
    price_dir="data/parquet/prices",
    features_path="data/parquet/features",
    n_sample=10,
    random_state=42,
)

# Ingest a sample of 10 tickers
dm.process_sample_files()

# Compute features and save
dm.featurize()
```

### Command-line interface

```bash
# Ingest all tickers then featurize
uv run python -m stock_prices.data_manager --action all

# Ingest a reproducible sample of 50 tickers only
uv run python -m stock_prices.data_manager --action ingest --n_sample 50 --random_state 0

# Featurize previously ingested parquet files
uv run python -m stock_prices.data_manager --action featurize
```
