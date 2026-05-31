"""
Shared pipeline factory and MLflow experiment runner for credit risk experiments.

Centralises three components used by all experiment scripts:
  - get_pipe()                  — sklearn preprocessing + LogisticRegression pipeline
  - get_or_create_experiment()  — MLflow experiment lookup / creation
  - run_cv()                    — cross-validation loop with full MLflow logging
"""

import os

import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

from utils.logger import get_logger

load_dotenv(override=True)
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

_logs = get_logger(__name__)
mlflow.set_tracking_uri(MLFLOW_URI)


def get_pipe() -> Pipeline:
    """Build and return the sklearn preprocessing + logistic regression pipeline.

    Two preprocessing branches run in parallel via ColumnTransformer:

    ``num_standard`` — count and age columns:
        SimpleImputer (median) → StandardScaler

    ``num_pow_cols`` — skewed continuous columns (utilization, income, debt ratio):
        SimpleImputer (median) → StandardScaler → PowerTransformer (Yeo-Johnson)

    Binary indicator columns created in ``data.load_data``
    (``high_debt_ratio``, ``missing_monthly_income``, ``missing_num_dependents``)
    pass through the ColumnTransformer unchanged via ``remainder='passthrough'``.

    Returns
    -------
    Pipeline
        Unfitted pipeline ready for ``set_params`` and ``fit``.
    """
    num_std_cols = [
        'num_30_59_days_late',
        'num_60_89_days_late',
        'num_90_days_late',
        'num_open_credit_loans',
        'num_real_estate_loans',
        'age',
        'num_dependents',
    ]
    num_pow_cols = [
        'revolving_unsecured_line_utilization',
        'monthly_income',
        'debt_ratio',
    ]

    preproc_std = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    preproc_power = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('power', PowerTransformer()),
    ])

    ct = ColumnTransformer(transformers=[
        ('num_standard', preproc_std, num_std_cols),
        ('num_pow_cols', preproc_power, num_pow_cols),
    ], remainder='passthrough')

    return Pipeline(steps=[
        ('preproc', ct),
        ('clf', LogisticRegression()),
    ])


def get_or_create_experiment(experiment_name: str) -> str:
    """Return the MLflow experiment ID, creating the experiment if it does not exist.

    Parameters
    ----------
    experiment_name : str
        Display name of the MLflow experiment.

    Returns
    -------
    str
        MLflow experiment ID.
    """
    _logs.info(f'Looking up MLflow experiment: {experiment_name}')
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        _logs.info(f'Found existing experiment (id={experiment.experiment_id})')
        return experiment.experiment_id
    _logs.info(f'Creating new MLflow experiment: {experiment_name}')
    return mlflow.create_experiment(experiment_name)


def run_cv(
    pipe: Pipeline,
    X: pd.DataFrame,
    Y: pd.Series,
    params: dict,
    folds: int = 5,
    experiment_name: str | None = None,
    model_name: str | None = None,
    test_size: float = 0.2,
    scoring: list[str] | str | None = None,
    random_state: int | None = None,
    tags: dict | None = None,
    nested: bool = False,
    log_model: bool = True,
) -> dict:
    """Run one cross-validated experiment and log everything to MLflow.

    Splits data into train/test, runs k-fold cross-validation on the training
    set, logs mean CV metrics, and — when ``log_model=True`` — fits the final
    pipeline on the full training set and logs the artifact.

    Parameters
    ----------
    pipe : Pipeline
        Unfitted sklearn pipeline (modified in place via ``set_params``).
    X : pd.DataFrame
        Feature matrix.
    Y : pd.Series
        Target vector.
    params : dict
        Pipeline parameters passed to ``pipe.set_params(**params)``.
    folds : int
        Number of cross-validation folds.
    experiment_name : str | None
        MLflow experiment name. If ``None``, the currently active experiment
        is used (caller is responsible for setting it beforehand).
    model_name : str | None
        If provided, the logged model is also registered under this name in
        the MLflow Model Registry.
    test_size : float
        Fraction of data held out for the test split.
    scoring : list[str] | str | None
        Scoring metrics for ``cross_validate``. Defaults to ``['neg_log_loss']``.
    random_state : int | None
        Seed for the train/test split.
    tags : dict | None
        MLflow tags attached to the run.
    nested : bool
        Start a nested child run. Set to ``True`` when called from inside an
        existing ``mlflow.start_run`` context (e.g. hyperopt trials).
    log_model : bool
        If ``True``, fit the pipeline on the training set after CV and log the
        fitted model as an MLflow artifact. Set to ``False`` for high-volume
        searches where per-trial model storage is not needed.

    Returns
    -------
    dict
        Mean cross-validation metrics (keys match sklearn's ``cross_validate``
        output, e.g. ``test_neg_log_loss``, ``train_accuracy``).

    Example
    -------
    >>> from credit.pipeline import get_pipe, run_cv
    >>> from credit.data import load_data
    >>> X, Y = load_data()
    >>> metrics = run_cv(get_pipe(), X, Y, {'clf__C': 0.5},
    ...                  experiment_name='my_experiment', random_state=42)
    """
    if scoring is None:
        scoring = ['neg_log_loss']
    if isinstance(scoring, str):
        scoring = [scoring]
    if tags is None:
        tags = {}

    if experiment_name is not None:
        experiment_id = get_or_create_experiment(experiment_name)
        mlflow.set_experiment(experiment_id=experiment_id)

    _logs.info(f'Starting CV run — experiment={experiment_name}, nested={nested}')

    with mlflow.start_run(nested=nested):
        mlflow.log_params(params)
        mlflow.log_params({
            'folds': folds,
            'test_size': test_size,
            'random_state': random_state,
            'scoring': scoring,
        })
        mlflow.set_tags(tags)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state
        )
        pipe.set_params(**params)
        res_cv = cross_validate(
            pipe, X_train, Y_train,
            cv=folds, scoring=scoring, return_train_score=True,
        )
        mean_res_cv = pd.DataFrame(res_cv).mean().to_dict()
        mlflow.log_metrics(mean_res_cv)
        _logs.info(f'CV metrics: {mean_res_cv}')

        if log_model:
            _logs.info('Fitting final model and logging artifact')
            pipe.fit(X_train, Y_train)
            signature = infer_signature(X_train, pipe.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=pipe,
                artifact_path='model',
                signature=signature,
                input_example=X_train.head(5),
                registered_model_name=model_name,
            )

    return mean_res_cv
