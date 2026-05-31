# credit

Logistic regression experiments on the Give Me Some Credit dataset, tracked with MLflow.

---

## Module structure

| File | Purpose |
|------|---------|
| `data.py` | Load and preprocess the raw CSV into `(X, Y)` |
| `logistic.py` | Logistic regression pipeline factory (`get_pipe`) |
| `experiment.py` | MLflow experiment helpers and CV runner |
| `exp__logistic_simple.py` | Single baseline run with fixed default parameters |
| `exp__logistic_grid_search.py` | Exhaustive grid search over a predefined parameter space |
| `exp__logistic_hyperopt.py` | Bayesian optimisation with Optuna (TPE); registers the best model |

---

## Environment variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `CREDIT_DATA` | `data.py` | Path to the raw Give Me Some Credit CSV file |
| `MLFLOW_TRACKING_URI` | `experiment.py` | MLflow tracking server URL (e.g. `http://localhost:5001`) |

---

## `data.py` — `load_data`

```python
load_data(file: str | None = CREDIT_FILE) -> tuple[pd.DataFrame, pd.Series]
```

Drops the unnamed index column, renames all columns to snake_case, engineers
three binary indicators, coerces everything to numeric, and returns `(X, Y)`.

**Engineered columns:**

| Column | Definition |
|--------|-----------|
| `high_debt_ratio` | 1 if `debt_ratio > 1`, else 0 |
| `missing_monthly_income` | 1 if `monthly_income` is NaN, else 0 |
| `missing_num_dependents` | 1 if `num_dependents` is NaN, else 0 |

---

## `logistic.py` — pipeline factory

### `get_pipe() -> Pipeline`

Builds a fresh unfitted sklearn pipeline with two parallel preprocessing branches:

| Branch | Columns | Steps |
|--------|---------|-------|
| `num_standard` | count and age columns | `SimpleImputer(median)` → `StandardScaler` |
| `num_pow_cols` | utilization, income, debt ratio | `SimpleImputer(median)` → `StandardScaler` → `PowerTransformer` |

Binary indicator columns pass through unchanged. Classifier: `LogisticRegression`.

---

## `experiment.py` — MLflow helpers and CV runner

### `get_or_create_experiment(experiment_name: str) -> str`

Returns the MLflow experiment ID, creating the experiment if it does not exist.

### `run_cv(pipe, X, Y, params, ...) -> dict`

Runs one cross-validated experiment and logs everything to MLflow.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pipe` | — | Unfitted sklearn pipeline |
| `X, Y` | — | Feature matrix and target vector |
| `params` | — | Passed to `pipe.set_params(**params)` |
| `folds` | `5` | CV folds |
| `experiment_name` | `None` | MLflow experiment name; `None` uses the active experiment |
| `model_name` | `None` | If set, registers the model in the MLflow Model Registry |
| `test_size` | `0.2` | Train/test split fraction |
| `scoring` | `['neg_log_loss']` | sklearn scoring metrics |
| `random_state` | `None` | Seed for the train/test split |
| `tags` | `{}` | MLflow run tags |
| `nested` | `False` | Start a nested child run (used by hyperopt trials) |
| `log_model` | `True` | Fit and log the model artifact after CV |

---

## Running the experiments

All commands are run from `05_src/`:

```bash
# Baseline — one run, model registered as CreditLogisticSimple
uv run python -m credit.exp__logistic_simple

# Grid search — one run per parameter combination, metrics only
uv run python -m credit.exp__logistic_grid_search

# Hyperopt — Bayesian optimisation, best model registered as CreditLogisticHyperopt
uv run python -m credit.exp__logistic_hyperopt
```

---

## MLflow experiment names and model logging strategy

| Script | MLflow experiment | Model logged? | Registered as |
|--------|------------------|--------------|---------------|
| `exp__logistic_simple` | `credit_single_run_logistic` | Yes — the single run | `CreditLogisticSimple` |
| `exp__logistic_grid_search` | `credit_grid_search_logistic` | No — metrics only per trial | — |
| `exp__logistic_hyperopt` | `credit_hyperopt_logistic` | Parent run only (best model) | `CreditLogisticHyperopt` |

For grid search, identify the best run in the MLflow UI and register manually,
or promote the result to the simple experiment for a clean registration run.
