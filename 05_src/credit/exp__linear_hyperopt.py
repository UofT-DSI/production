"""
Logistic regression hyperparameter search over penalty type and regularisation strength.

Explores l1, l2, and elasticnet penalties with solver-compatible combinations via
Optuna TPE. Each trial is a nested child MLflow run (metrics only). After optimisation,
the parent run logs the best parameters, fits a final model on the training split,
and registers it as CreditLinear in the MLflow Model Registry.
"""

import mlflow
import mlflow.sklearn
import optuna
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split

from credit.data import load_data
from credit.experiment import get_or_create_experiment, run_cv
from credit.linear import get_pipe
from utils.logger import get_logger

_logs = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def linear_search(
    scoring: list[str] | None = None,
    folds: int = 5,
    random_state: int = 42,
    n_trials: int = 50,
    test_size: float = 0.2,
    experiment_name: str = 'credit_linear_optuna',
    model_name: str = 'CreditLinear',
) -> None:
    """Search over logistic regression hyperparameters using Optuna TPE.

    The search space covers penalty type (l1, l2, elasticnet), regularisation
    strength C, class weighting, and — for l2 — solver choice. Solver is fixed
    to saga for l1 and elasticnet as it is the only compatible option.

    Each trial is logged as a nested child MLflow run (metrics only). After
    ``study.optimize`` completes, the parent run logs the best decoded parameters
    and best log-loss, then fits a final model and registers it.

    Parameters
    ----------
    scoring : list[str] | None
        Scoring metrics for ``cross_validate``. Defaults to ``['neg_log_loss']``.
    folds : int
        Number of cross-validation folds per trial.
    random_state : int
        Seed for the train/test split and the Optuna TPE sampler.
    n_trials : int
        Number of Optuna trials.
    test_size : float
        Fraction of data held out for the final test split.
    experiment_name : str
        MLflow experiment name.
    model_name : str
        Name under which the best model is registered in the MLflow Model Registry.
    """
    if scoring is None:
        scoring = ['neg_log_loss']

    _logs.info(f'Starting linear search: n_trials={n_trials}')
    X, Y = load_data()
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    experiment_id = get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    with mlflow.start_run():

        def objective(trial: optuna.Trial) -> float:
            penalty = trial.suggest_categorical('clf__penalty', ['l1', 'l2', 'elasticnet'])

            if penalty == 'elasticnet':
                solver = 'saga'
                extra = {'clf__l1_ratio': trial.suggest_float('clf__l1_ratio', 0.0, 1.0)}
            elif penalty == 'l1':
                solver = 'saga'
                extra = {'clf__l1_ratio': 1.0}
            else:
                solver = trial.suggest_categorical('clf__solver', ['lbfgs', 'saga'])
                extra = {'clf__l1_ratio': 0.0}

            params = {
                'clf__C': trial.suggest_float('clf__C', 1e-4, 100.0, log=True),
                'clf__solver': solver,
                'clf__class_weight': trial.suggest_categorical(
                    'clf__class_weight', [None, 'balanced']
                ),
                'clf__random_state': random_state,
                **extra,
            }
            metrics = run_cv(
                get_pipe(), X, Y, params,
                folds=folds,
                scoring=scoring,
                random_state=random_state,
                tags={'optimizer': 'optuna', 'model_family': 'linear'},
                nested=True,
                log_model=False,
            )
            return -metrics['test_neg_log_loss']

        sampler = optuna.samplers.TPESampler(seed=random_state)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials)

        # Reconstruct full params for logging and final fit.
        # clf__solver is only in best_params when penalty='l2'; otherwise saga was hardcoded.
        best = study.best_params
        penalty = best['clf__penalty']
        best_params = {
            'clf__C': best['clf__C'],
            'clf__solver': best.get('clf__solver', 'saga'),
            'clf__class_weight': best['clf__class_weight'],
            'clf__random_state': random_state,
        }
        if penalty == 'elasticnet':
            best_params['clf__l1_ratio'] = best['clf__l1_ratio']

        best_loss = study.best_value
        _logs.info(f'Best params: {best_params}')
        _logs.info(f'Best test_neg_log_loss: {-best_loss:.4f}')

        mlflow.log_params(best_params)
        mlflow.log_metric('best_test_neg_log_loss', -best_loss)

        _logs.info(f'Fitting best model and registering as {model_name}')
        best_pipe = get_pipe()
        best_pipe.set_params(**best_params)
        best_pipe.fit(X_train, Y_train)
        signature = infer_signature(X_train, best_pipe.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=best_pipe,
            name='best_model',
            signature=signature,
            input_example=X_train.head(5),
            registered_model_name=model_name,
        )


if __name__ == '__main__':
    linear_search()
