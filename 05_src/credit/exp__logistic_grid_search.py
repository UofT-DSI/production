"""
Grid search over logistic regression hyperparameters.

Iterates over the full Cartesian product of the parameter space, logging
CV metrics for each combination to MLflow under the
``credit_grid_search_logistic`` experiment.

Model artifacts are not logged per trial (use the MLflow UI to identify the
best run, then register via ``exp__logistic_simple.py`` or the registry UI).
"""

import mlflow
from sklearn.model_selection import ParameterGrid

from credit.data import load_data
from credit.pipeline import get_or_create_experiment, get_pipe, run_cv
from utils.logger import get_logger

_logs = get_logger(__name__)


def grid_search(
    scoring: list[str] | None = None,
    folds: int = 5,
    random_state: int = 42,
) -> None:
    """Exhaustively search a predefined hyperparameter grid.

    Each combination is logged as a separate MLflow run under a shared
    experiment so results can be compared in the tracking UI.
    A fresh pipeline is built for every combination to prevent parameter
    bleed between iterations.

    Parameters
    ----------
    scoring : list[str] | None
        Scoring metrics for ``cross_validate``.
        Defaults to ``['neg_log_loss', 'balanced_accuracy', 'f1']``.
    folds : int
        Number of cross-validation folds.
    random_state : int
        Seed for the train/test split and the logistic regression solver.
    """
    if scoring is None:
        scoring = ['neg_log_loss', 'balanced_accuracy', 'f1']

    X, Y = load_data()

    param_space = {
        'preproc__num_standard__imputer__add_indicator': [False, True],
        'clf__C': [0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
        'clf__class_weight': [None, 'balanced'],
        'clf__penalty': ['l2'],
        'clf__random_state': [random_state],
        'clf__solver': ['lbfgs', 'liblinear'],
    }
    param_grid = list(ParameterGrid(param_space))
    _logs.info(f'Grid search: {len(param_grid)} combinations')

    experiment_id = get_or_create_experiment('credit_grid_search_logistic')
    mlflow.set_experiment(experiment_id=experiment_id)

    for k, params in enumerate(param_grid):
        _logs.info(f'Grid search iteration {k + 1}/{len(param_grid)}')
        run_cv(
            get_pipe(), X, Y, params,
            folds=folds,
            scoring=scoring,
            random_state=random_state,
            tags={'optimizer': 'grid_search'},
            log_model=False,
        )


if __name__ == "__main__":
    grid_search()
