# credit

Logistic regression experiments on the Give Me Some Credit dataset, tracked with MLflow.

---

## Module structure

| File | Purpose |
|------|---------|
| `data.py` | Load the raw CSV into `(X, Y)` |
| `logistic.py` | Baseline pipeline factory (`get_pipe`) |
| `linear.py` | Extended pipeline factory with per-variable feature engineering |
| `experiment.py` | MLflow experiment helpers and CV runner |
| `exp__logistic_simple.py` | Single baseline run with fixed default parameters |
| `exp__logistic_grid_search.py` | Exhaustive grid search over a predefined parameter space |
| `exp__logistic_hyperopt.py` | Optuna search over default hyperparameters; registers the best model |
| `exp__linear.py` | Optuna search over penalty type, regularisation strength, and solver |

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

Drops the unnamed index column, renames all columns to snake_case, coerces
everything to numeric, and returns `(X, Y)`. No feature engineering is applied;
raw columns are passed through as-is.

---

## `logistic.py` — baseline pipeline factory

### `get_pipe() -> Pipeline`

Builds a fresh unfitted sklearn pipeline with two parallel preprocessing branches:

| Branch | Columns | Steps |
|--------|---------|-------|
| `num_standard` | count and age columns | `SimpleImputer(median)` → `StandardScaler` |
| `num_pow_cols` | utilization, income, debt ratio | `SimpleImputer(median)` → `StandardScaler` → `PowerTransformer` |

Classifier: `LogisticRegression`.

---

## `linear.py` — extended pipeline factory

### `get_pipe() -> Pipeline`

Builds a fresh unfitted pipeline with per-variable transformations chosen to
address the known data-quality and distributional issues in this dataset:

| Branch | Columns | Steps |
|--------|---------|-------|
| `late` | num_*_days_late (×3) | clip [0, 15] → `SimpleImputer(median)` → `StandardScaler` |
| `util` | revolving_unsecured_line_utilization | clip [0, 2] → `log1p` → `StandardScaler` |
| `debt` | debt_ratio | clip [0, 5] → `log1p` → `StandardScaler` |
| `income` | monthly_income | `SimpleImputer(median)` → `log1p` → `StandardScaler` |
| `income_ind` | monthly_income | `MissingIndicator` (binary, unscaled) |
| `age` | age | clip [18, 105] → `StandardScaler` |
| `open_loans` | num_open_credit_loans | clip [0, 30] → `StandardScaler` |
| `real_estate` | num_real_estate_loans | clip [0, 10] → `StandardScaler` |
| `dep` | num_dependents | clip [0, 10] → `SimpleImputer(median)` → `StandardScaler` |
| `dep_ind` | num_dependents | `MissingIndicator` (binary, unscaled) |

Clipping thresholds for late-payment counts address values of 96 and 98, which
are widely treated as coding errors in this dataset. Utilization and debt ratio
are log-compressed after capping to reduce the influence of extreme outliers on
the linear decision boundary. Missing indicators for income and dependents are
kept unscaled so the missingness signal is not distorted.

Classifier: `LogisticRegression(max_iter=1000, solver='saga')`. `saga` is the
default because it is the only solver compatible with all three penalty types
(`l1`, `l2`, `elasticnet`), keeping penalty as a live hyperparameter.

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
| `nested` | `False` | Start a nested child run (used by Optuna trials) |
| `log_model` | `True` | Fit and log the model artifact after CV |

---

## Running the experiments

All commands are run from `05_src/`:

```bash
# Baseline — one run, model registered as CreditLogisticSimple
uv run python -m credit.exp__logistic_simple

# Grid search — one run per parameter combination, metrics only
uv run python -m credit.exp__logistic_grid_search

# Optuna (baseline pipeline) — Bayesian optimisation, best model registered as CreditLogisticHyperopt
uv run python -m credit.exp__logistic_hyperopt

# Optuna (extended pipeline) — searches penalty type, C, and solver; registers CreditLinear
uv run python -m credit.exp__linear
```

---

## MLflow experiment names and model logging strategy

| Script | MLflow experiment | Model logged? | Registered as |
|--------|------------------|--------------|---------------|
| `exp__logistic_simple` | `credit_single_run_logistic` | Yes — the single run | `CreditLogisticSimple` |
| `exp__logistic_grid_search` | `credit_grid_search_logistic` | No — metrics only per trial | — |
| `exp__logistic_hyperopt` | `credit_hyperopt_logistic` | Parent run only (best model) | `CreditLogisticHyperopt` |
| `exp__linear` | `credit_linear_optuna` | Parent run only (best model) | `CreditLinear` |

For grid search, identify the best run in the MLflow UI and register manually,
or promote the result to the simple experiment for a clean registration run.
