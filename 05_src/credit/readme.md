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
| `exp__linear_hyperopt.py` | Optuna search over penalty type, regularisation strength, and solver |

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

```python
from credit.data import load_data
from credit.logistic import get_pipe

X, Y = load_data()
pipe = get_pipe()
pipe.set_params(clf__C=0.5, clf__l1_ratio=0.0)
pipe.fit(X, Y)
pipe.predict_proba(X.head())
```

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
| `income_ind` | monthly_income | `MissingIndicator(features='all')` (binary, unscaled) |
| `age` | age | clip [18, 105] → `StandardScaler` |
| `open_loans` | num_open_credit_loans | clip [0, 30] → `StandardScaler` |
| `real_estate` | num_real_estate_loans | clip [0, 10] → `StandardScaler` |
| `dep` | num_dependents | clip [0, 10] → `SimpleImputer(median)` → `StandardScaler` |
| `dep_ind` | num_dependents | `MissingIndicator(features='all')` (binary, unscaled) |

Clipping thresholds for late-payment counts address values of 96 and 98, which
are widely treated as coding errors in this dataset. Utilization and debt ratio
are log-compressed after capping to reduce the influence of extreme outliers on
the linear decision boundary. Missing indicators for income and dependents are
kept unscaled so the missingness signal is not distorted.

Classifier: `LogisticRegression(max_iter=1000, solver='saga')`. `saga` is the
default because it is the only solver compatible with all three penalty types
(`l1`, `l2`, `elasticnet`), keeping penalty as a live hyperparameter.

Missing indicators use `features='all'` so each always outputs one column, even
when a cross-validation fold has no missing values in the training part.

```python
from credit.data import load_data
from credit.linear import get_pipe

X, Y = load_data()
pipe = get_pipe()
pipe.set_params(clf__C=0.1, clf__l1_ratio=1.0)  # l1 penalty
pipe.fit(X, Y)
pipe.named_steps['preproc'].transform(X.head()).shape  # (5, 12)
```

---

## Penalty selection via `l1_ratio`

All experiments set the penalty through `clf__l1_ratio` only, which requires
scikit-learn >= 1.8 (the `penalty` argument is deprecated there):

| `clf__l1_ratio` | Penalty |
|-----------------|---------|
| `0.0` (default) | l2 |
| `1.0` | l1 |
| between 0 and 1 | elasticnet |

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

Returns mean CV metrics keyed as in `cross_validate` (e.g. `test_neg_log_loss`).
Requires `CREDIT_DATA` and a running MLflow server at `MLFLOW_TRACKING_URI`.

### `SKOPS_TRUSTED_TYPES`

MLflow saves sklearn models with [skops](https://skops.readthedocs.io/), which refuses
to load any type not marked as trusted. Every `log_model` call in this package passes
`SKOPS_TRUSTED_TYPES = ['numpy.clip', 'numpy.dtype']` — the non-default types the two
credit pipelines contain — so registered models load with a plain
`mlflow.sklearn.load_model('models:/<name>/<version>')`. Extend the list if a pipeline
gains another custom function or type.

```python
from credit.data import load_data
from credit.experiment import run_cv
from credit.logistic import get_pipe

X, Y = load_data()
metrics = run_cv(get_pipe(), X, Y, {'clf__C': 0.5},
                 experiment_name='my_experiment', random_state=42)
metrics['test_neg_log_loss']
```

---

## `exp__linear_hyperopt.py` — linear search

### `suggest_params(trial: optuna.trial.BaseTrial, random_state: int) -> dict`

Samples one parameter set for the extended pipeline. Each Optuna trial stores
the returned dict as a user attribute, and the final model is refit from the
best trial's stored dict, so logged and registered parameters match exactly.

| Penalty | `clf__solver` | `clf__l1_ratio` |
|---------|---------------|-----------------|
| `l1` | `saga` | `1.0` |
| `l2` | `lbfgs` or `saga` (searched) | `0.0` |
| `elasticnet` | `saga` | searched in [0, 1] |

`clf__C` is searched log-uniformly in [1e-4, 100] and `clf__class_weight` in
`{None, 'balanced'}`.

```python
from optuna.trial import FixedTrial
from credit.exp__linear_hyperopt import suggest_params

trial = FixedTrial({'clf__penalty': 'l1', 'clf__C': 0.1, 'clf__class_weight': None})
suggest_params(trial, random_state=42)
# {'clf__C': 0.1, 'clf__solver': 'saga', 'clf__class_weight': None,
#  'clf__l1_ratio': 1.0, 'clf__random_state': 42}
```

### `linear_search(scoring=None, folds=5, random_state=42, n_trials=50, ...) -> None`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scoring` | `['neg_log_loss']` | sklearn scoring metrics |
| `folds` | `5` | CV folds per trial |
| `random_state` | `42` | Seed for the split, classifier, and TPE sampler |
| `n_trials` | `50` | Number of Optuna trials |
| `test_size` | `0.2` | Fraction held out for the final test split |
| `experiment_name` | `'credit_linear_optuna'` | MLflow experiment name |
| `model_name` | `'CreditLinear'` | Registered model name |

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
uv run python -m credit.exp__linear_hyperopt
```

---

## MLflow experiment names and model logging strategy

| Script | MLflow experiment | Model logged? | Registered as |
|--------|------------------|--------------|---------------|
| `exp__logistic_simple` | `credit_single_run_logistic` | Yes — the single run | `CreditLogisticSimple` |
| `exp__logistic_grid_search` | `credit_grid_search_logistic` | No — metrics only per trial | — |
| `exp__logistic_hyperopt` | `credit_hyperopt_logistic` | Parent run only (best model) | `CreditLogisticHyperopt` |
| `exp__linear_hyperopt` | `credit_linear_optuna` | Parent run only (best model) | `CreditLinear` |

For grid search, identify the best run in the MLflow UI and register manually,
or promote the result to the simple experiment for a clean registration run.
