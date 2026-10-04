---
title: Testing the 05_src ML code against real data with an isolated MLflow store
date: 2026-10-03
category: best-practices
module: 05_src/tests
problem_type: best_practice
component: testing_framework
severity: medium
applies_when:
  - "Adding or changing tests for 05_src/credit, 05_src/stock_prices, or 05_src/utils"
  - "Adding a new experiment script that logs or registers models in MLflow"
  - "Changing import-time behaviour (load_dotenv, mlflow.set_tracking_uri, module-level env reads) in 05_src"
  - "Upgrading pandas, dask, mlflow, or pytest"
tags: [pytest, mlflow, real-data, test-isolation, dask, pandas-3, skops, precision-tests]
---

# Testing the 05_src ML code against real data with an isolated MLflow store

## Context

`05_src` had no tests until PR #184. The code is course material: data pipelines (`stock_prices`), MLflow-logged experiments (`credit`), and a logger (`utils`). Three properties of the code make testing harder than usual:

- Modules do work at import time. `05_src/credit/experiment.py` runs `load_dotenv(override=True)` and `mlflow.set_tracking_uri(...)` on import. `05_src/credit/data.py` and `05_src/stock_prices/data_manager.py` read their data paths from the environment into module constants.
- `05_src/.env` uses paths relative to `05_src` (for example `CREDIT_DATA=../05_src/data/credit/cs-training.csv`). They only resolve when the working directory is `05_src`.
- The environment also installs an unrelated PyPI package named `utils`. It can shadow the local `05_src/utils`.

The suite was built against the real (gitignored) course data. The design was a deliberate choice for fidelity over CI portability.

## Guidance

**Shape of the suite**

- Tests live in `05_src/tests/`, mirroring the source packages (`05_src/credit`, `05_src/stock_prices`, `05_src/utils`).
- Each test carries exactly one layer marker: `unit` (exact values), `pipeline` (data pipeline end to end, compared against an independent pandas reference), or `experiment` (MLflow-logging runs). `05_src/tests/layer_markers.py` aborts collection with a usage error when a test has zero or several layer markers.
- `pyproject.toml` sets `testpaths = ["05_src/tests"]`, `pythonpath = ["05_src"]` (so the local `utils` wins), `--import-mode=importlib`, and `--strict-markers`.

**Isolation order matters (`05_src/tests/harness.py`)**

1. Create one session temp root and set `LOG_DIR` before importing any source module (`harness.py:24-26`).
2. Import `credit.experiment` first, then re-apply `LOG_DIR` and redirect MLflow to a SQLite store in the temp root. The import's `load_dotenv(override=True)` resets both (`harness.py:29-39`).
3. Set `_MLFLOW_SERVER_ARTIFACT_ROOT` to the temp root, so experiments the tests did not pre-create still keep their artifacts out of `05_src/mlruns` (`harness.py:37`).
4. Change into `05_src` in a session-scoped autouse fixture (`conftest.py:23-25`), **not** at conftest import time.

**Missing data fails, never skips.** `require_data_path` (`harness.py:42`) calls `pytest.fail` and names the variable and `05_src/tests/readme.md`. A partly green run would hide a broken setup.

**Assert exact values.** Compute the expected value independently and compare it with `pytest.approx`. Examples: hand-built frames for clip/log bounds, `cross_validate` re-run for `run_cv` metrics, a pandas sort-then-shift reference for lags, and the best child run's logged params as the oracle for the refit.

**Prove regression tests can fail.** For each fix, temporarily revert it and watch the test go red. The lag tests and the forced-l1 refit test were both checked this way.

## Why This Matters

The three real defects this suite found would all have passed smoke tests:

| Defect | Why a smoke test misses it |
|---|---|
| `DataManager.create_features` shifted `Close` on the unsorted group, then assigned by position onto the date-sorted frame. 78-100% of `Close_lag_1` / `Returns` values were wrong. | The features exist, have the right shape, and are finite. Only a comparison against an independent reference shows they are wrong. |
| Under pandas 3 the hand-built Dask `meta` became `str`-typed, so `featurize` raised `TypeError`. | Caught by any pipeline run, but only once the pipeline is actually executed end to end. |
| MLflow 3.16 saves sklearn models with skops, which refuses `numpy.dtype` / `numpy.clip` on load. No registered credit model could be loaded back. | Logging and registration succeed. Only loading the registered model fails. |

The fixes are `_add_close_lag` plus a meta derived from the loaded frame (`05_src/stock_prices/data_manager.py:212`, `:226`), and `SKOPS_TRUSTED_TYPES` passed at every `log_model` call (`05_src/credit/experiment.py:26`).

Two isolation bugs also came up while building the harness:

- **An import-time `os.chdir` made pytest collect all of `05_src`.** Pytest resolves `testpaths` after it loads `conftest.py`, so the early directory change redirected collection. That imported `05_src/experiment_tracking/test_mlflow.py`, which called `mlflow.set_tracking_uri('http://localhost:5001')` mid-session and silently pointed tests at the class server. It appeared only in the full run, not when files ran one at a time. It was found by wrapping `mlflow.set_tracking_uri` in a throwaway tracing plugin.
- **Importing a module twice re-runs its setup.** Importing `tests.conftest` from a test module would execute conftest a second time, creating a second temp root and re-importing. Shared state therefore lives in `05_src/tests/harness.py`, which both conftest and tests import.

## When to Apply

- Any new test under `05_src/tests`. Reuse `05_src/tests/helpers.py` (`unique_name`, `experiment_runs`, `assert_valid_probabilities`) and mark the test with one layer.
- Any new `mlflow.sklearn.log_model` call. Pass `skops_trusted_types=SKOPS_TRUSTED_TYPES`, and extend the list if the pipeline gains a new custom function or type.
- Any change to import-time code in `05_src`. Re-run the full suite, not single files, because collection-order effects only show up then.
- Any dependency upgrade. Pandas 3 and MLflow 3.16 each broke previously working code silently.

## Examples

Run the suite from the repository root:

```bash
uv run python -m pytest                      # everything (~70 s with data cached)
uv run python -m pytest -m unit              # one layer
uv run python -m pytest -m "not experiment"  # skip the slow layer
```

Precision over smoke — the pattern `05_src/tests/stock_prices/test_pipeline.py` follows (simplified):

```python
# Smoke (passes on the buggy lag):  assert features['Close_lag_1'].notna().any()
# Precision (fails on the buggy lag):
expected = df.sort_values('Date').assign(Close_lag_1=lambda d: d['Close'].shift(1))
np.testing.assert_allclose(got['Close_lag_1'], expected['Close_lag_1'], equal_nan=True)
```

Tracing who changed the MLflow URI (throwaway plugin, loaded with `PYTHONPATH=<dir> pytest -p trace_uri`):

```python
import traceback, mlflow
_original = mlflow.set_tracking_uri
def _traced(uri):
    print('SET', uri, ''.join(traceback.format_stack(limit=12)))
    return _original(uri)
mlflow.set_tracking_uri = _traced
```

## Related

- `05_src/tests/readme.md` — student-facing setup and layer guide
- `docs/plans/2026-10-03-1928-feat-src-test-suite-plan.md` — requirements and decisions
