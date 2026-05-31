"""
Baseline logistic regression experiment with a fixed set of default hyperparameters.

Runs a single cross-validated experiment and logs params, CV metrics, and
the fitted model to MLflow under the ``credit_single_run_logistic`` experiment.
The model is registered in the MLflow Model Registry as ``CreditLogisticSimple``.
"""

from credit.data import load_data
from credit.pipeline import get_pipe, run_cv
from utils.logger import get_logger

_logs = get_logger(__name__)


def single_run(
    scoring: list[str] | None = None,
    folds: int = 5,
    random_state: int = 42,
) -> None:
    """Run one cross-validated logistic regression experiment.

    All pipeline parameters are fixed at their defaults and logged explicitly
    so every param is visible in the MLflow UI for comparison with other
    experiment types.

    Parameters
    ----------
    scoring : list[str] | None
        Scoring metrics for ``cross_validate``.
        Defaults to ``['neg_log_loss', 'accuracy', 'f1']``.
    folds : int
        Number of cross-validation folds.
    random_state : int
        Seed for the train/test split and the logistic regression solver.
    """
    if scoring is None:
        scoring = ['neg_log_loss', 'accuracy', 'f1']

    _logs.info('Starting single logistic regression run')
    pipe = get_pipe()
    X, Y = load_data()

    params = {
        'preproc__num_standard__imputer__add_indicator': False,
        'preproc__num_standard__imputer__copy': True,
        'preproc__num_standard__imputer__strategy': 'median',
        'preproc__num_standard__scaler__with_mean': True,
        'preproc__num_standard__scaler__with_std': True,
        'preproc__num_pow_cols__imputer__strategy': 'median',
        'preproc__num_pow_cols__scaler__with_mean': True,
        'preproc__num_pow_cols__scaler__with_std': True,
        'preproc__num_pow_cols__power__method': 'yeo-johnson',
        'preproc__num_pow_cols__power__standardize': True,
        'clf__C': 1.0,
        'clf__class_weight': None,
        'clf__fit_intercept': True,
        'clf__intercept_scaling': 1,
        'clf__l1_ratio': None,
        'clf__max_iter': 100,
        'clf__penalty': 'l2',
        'clf__random_state': random_state,
        'clf__solver': 'lbfgs',
    }

    run_cv(
        pipe, X, Y, params,
        folds=folds,
        experiment_name='credit_single_run_logistic',
        model_name='CreditLogisticSimple',
        test_size=0.2,
        scoring=scoring,
        random_state=random_state,
        tags={'optimizer': 'none'},
        log_model=True,
    )


if __name__ == "__main__":
    single_run()
