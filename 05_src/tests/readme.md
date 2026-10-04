# tests

The test suite for `credit`, `stock_prices`, and `utils`. It runs locally against the **real** course data, and it is written to be read: each test name states the behaviour it checks, and assertions compare against exact expected values rather than "is it non-empty?".

---

## Three layers

Every test carries exactly one layer marker. A test with none, or with two, stops the run before anything executes.

| Marker | What it proves | Example | Runtime* |
|--------|----------------|---------|----------|
| `unit` | One function or class returns exactly the right values. Hand-built inputs where the answer can be computed by hand; real data where the shape of the real file matters. | Late-payment counts of 96/98 are clipped to 15 in the linear pipeline; the logger never adds duplicate handlers | ~5 s |
| `pipeline` | A whole data pipeline runs end to end and every output value matches an independent pandas reference. | Real price CSVs → per-year parquet → `Close_lag_1` / `Returns` | ~10 s |
| `experiment` | Experiments run to completion and MLflow records the right runs, params, metrics, and loadable models. | All four `exp__*` entry points on the full credit data | 1–7 min |

\*Measured on the instructor's Windows laptop with the data cached. The experiment layer fits models on all 150,000 credit rows, so it can take several minutes when memory is tight. Use `-m "not experiment"` for a fast check.

The layers mirror how ML systems fail: a wrong transformation (unit), a pipeline that silently scrambles rows (pipeline), or an experiment that "runs" but logs or registers the wrong thing (experiment).

---

## Running the tests

Run from the **repository root**:

```bash
uv run python -m pytest                    # everything
uv run python -m pytest -m unit            # one layer
uv run python -m pytest -m "not experiment"  # skip the slow layer
uv run python -m pytest 05_src/tests/stock_prices -v   # one module
```

`pytest` is a dev dependency; `uv sync` installs it. Configuration lives in `[tool.pytest.ini_options]` in `pyproject.toml`.

---

## Data and environment setup

Data-dependent tests **fail** (they never skip) when the data is missing, naming the variable or file to fix. This is deliberate: a partly green run would hide a broken setup.

### 1. Download the data

| Dataset | Source | Put it in |
|---------|--------|-----------|
| Give Me Some Credit | [Kaggle competition data](https://www.kaggle.com/c/GiveMeSomeCredit/data) (free account required) | `05_src/data/credit/cs-training.csv` |
| Stock Market Dataset (Yahoo Finance prices) | [Kaggle dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset), or the [course Google Drive mirror](https://drive.google.com/drive/folders/1AA4gapDLpI194TGce1bY25sd91Km-tU3?usp=drive_link) | `05_src/data/prices_csv/stocks/*.csv` and `05_src/data/prices_csv/symbols_valid_meta.csv` |

Both are gitignored; nothing in `05_src/data` is committed.

### 2. Configure `05_src/.env`

The tests read the same variables as the course scripts. Paths are relative to `05_src`, which the test harness uses as its working directory.

| Variable | Used by | Example |
|----------|---------|---------|
| `CREDIT_DATA` | credit tests | `../05_src/data/credit/cs-training.csv` |
| `PRICE_CSV_DATA` | stock price pipeline tests (reads `<PRICE_CSV_DATA>/stocks/`) | `../05_src/data/prices_csv/` |

`LOG_DIR` and `MLFLOW_TRACKING_URI` may be set too, but the tests override both (see below). The MLflow Docker server does **not** need to be running.

---

## What the harness isolates

`tests/harness.py` runs before any source module is imported and redirects logs and MLflow into one temporary folder, which is deleted at the end of the session. A session fixture in `conftest.py` then switches the working directory to `05_src`. That switch must not happen at import time: pytest resolves `testpaths` after loading `conftest.py`, so an early switch makes it collect all of `05_src`.

| Concern | Where it goes during tests |
|---------|---------------------------|
| Working directory | `05_src` (so `.env` paths resolve like they do for the course scripts) |
| Log files | `<temp>/logs`, never `07_logs` |
| MLflow tracking + Model Registry | SQLite database in `<temp>`; the class server and its registry are untouched |
| MLflow artifacts and registered models | `<temp>/artifacts` |
| Pipeline outputs (parquet, features) | pytest's `tmp_path`; `05_src/data` is only read |

On Windows an empty `dsi_tests_*` folder may occasionally remain in your temp directory; it is safe to delete.

### Shared fixtures (`conftest.py`)

| Fixture | Scope | Returns |
|---------|-------|---------|
| `credit_csv` | session | `Path` to the credit CSV (fails loudly if missing) |
| `price_csv_dir` | session | `Path` to `<PRICE_CSV_DATA>/stocks` (fails loudly if missing) |
| `credit_xy` | session | `(X, Y)` from `load_data`, loaded once |
| `credit_sample` | session | a fixed-seed 5,000-row sample of `(X, Y)` for fast fits |
| `mlflow_client` | function | an `MlflowClient` bound to the temporary store |

```python
import pytest

pytestmark = pytest.mark.unit


def test_target_is_binary(credit_xy):
    _, Y = credit_xy
    assert set(Y.unique()) == {0, 1}
```

---

## Writing a new test

1. Put it next to the module it covers (`tests/credit/`, `tests/stock_prices/`, `tests/utils/`).
2. Mark it with exactly one layer, usually with a module-level `pytestmark = pytest.mark.unit`.
3. Assert exact values: compute the expected answer independently and compare with `pytest.approx` for floats.
4. Write outputs to `tmp_path`, never to `05_src/data`.
5. Reuse `tests/helpers.py`: `unique_name(prefix)` for MLflow experiment and model names, `experiment_runs(name)` to read an experiment's runs back, and `assert_valid_probabilities(model, X)` for classifier output.

---

## Gotchas

- **Two `utils` packages.** The project environment also installs an unrelated PyPI package called `utils`. Pytest's `pythonpath = ["05_src"]` puts the course code first, the same trick the lab notebooks use with `sys.path.insert(0, ...)`. `test_harness.py` checks that the local package wins.
- **`credit.experiment` resets the environment on import.** It calls `load_dotenv(override=True)`, which puts `.env`'s `LOG_DIR` and `MLFLOW_TRACKING_URI` back. The harness imports it first and then re-applies the redirects.
- **Experiment names are shared within a session.** `single_run` and `grid_search` use fixed experiment names, so their tests read runs from those experiments in the temporary store. Use unique names (for example with `uuid`) in new experiment tests.
