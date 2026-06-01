"""
Bayesian hyperparameter optimisation using Optuna (Tree-structured Parzen Estimator).

Each trial is a nested child MLflow run. After optimisation, the parent run
logs the best parameters, the best ``neg_log_loss``, fits a final model on
the training split, and registers it in the MLflow Model Registry as
``CreditLogisticHyperopt``.
"""

import mlflow
import mlflow.sklearn
import optuna
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split

from credit.data import load_data
from credit.experiment import get_or_create_experiment, run_cv
from credit.logistic import get_pipe
from utils.logger import get_logger

_logs = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def hyperparam_opt(
    scoring: list[str] | None = None,
    folds: int = 5,
    random_state: int = 42,
    max_evals: int = 10,
    test_size: float = 0.2,
    experiment_name: str = 'credit_hyperopt_logistic',
    model_name: str = 'CreditLogisticHyperopt',
) -> None:
    """Run Bayesian hyperparameter optimisation over the logistic regression pipeline.

    Each evaluation is logged as a nested child MLflow run (metrics only, no
    model artifact). After ``optimize`` completes, the parent run logs the best
    parameters and best ``neg_log_loss``, then fits a final pipeline on the
    training data and registers it in the Model Registry.

    Parameters
    ----------
    scoring : list[str] | None
        Scoring metrics for ``cross_validate`` in each trial.
        Defaults to ``['neg_log_loss', 'balanced_accuracy', 'f1']``.
    folds : int
        Number of cross-validation folds per trial.
    random_state : int
        Seed for the train/test split, the logistic regression solver, and the
        Optuna TPE sampler.
    max_evals : int
        Number of Optuna trials.
    test_size : float
        Fraction of data held out for the final test split.
    experiment_name : str
        MLflow experiment name.
    model_name : str
        Name under which the best model is registered in the MLflow Model Registry.
    """
    if scoring is None:
        scoring = ['neg_log_loss', 'balanced_accuracy', 'f1']

    _logs.info(f'Starting Optuna optimisation: max_evals={max_evals}')
    X, Y = load_data()
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    experiment_id = get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    with mlflow.start_run():

        def objective(trial: optuna.Trial) -> float:
            params = {
                'preproc__num_standard__imputer__add_indicator': trial.suggest_categorical(
                    'preproc__num_standard__imputer__add_indicator', [False, True]
                ),
                'clf__C': trial.suggest_float('clf__C', 0.1, 1.0),
                'clf__class_weight': trial.suggest_categorical('clf__class_weight', [None, 'balanced']),
                'clf__l1_ratio': 0.0,
                'clf__random_state': random_state,
            }
            metrics = run_cv(
                get_pipe(), X, Y, params,
                folds=folds,
                scoring=scoring,
                random_state=random_state,
                tags={'optimizer': 'optuna'},
                nested=True,
                log_model=False,
            )
            return -metrics['test_neg_log_loss']

        sampler = optuna.samplers.TPESampler(seed=random_state)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=max_evals)

        best_params = {
            **study.best_params,
            'clf__l1_ratio': 0.0,
            'clf__random_state': random_state,
        }
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


if __name__ == "__main__":
    hyperparam_opt()
