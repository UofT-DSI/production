"""
Integration tests for ``credit.experiment``: what ``run_cv`` and
``get_or_create_experiment`` actually record in MLflow.

Runs use a 5,000-row fixed-seed sample of the real credit data and the
session's temporary MLflow store (see ``tests/harness.py``).
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import pytest
from credit.experiment import get_or_create_experiment, run_cv
from credit.logistic import get_pipe
from sklearn.model_selection import cross_validate, train_test_split
from tests.helpers import assert_valid_probabilities, experiment_runs, unique_name

pytestmark = pytest.mark.experiment

PARAMS = {'clf__C': 0.5, 'clf__random_state': 0}
FOLDS = 3
TEST_SIZE = 0.2
SPLIT_SEED = 1


def _expected_cv_means(X, Y, scoring) -> dict:
    """Recompute run_cv's CV means independently, with the same split and folds."""
    X_train, _, Y_train, _ = train_test_split(X, Y, test_size=TEST_SIZE, random_state=SPLIT_SEED)
    pipe = get_pipe().set_params(**PARAMS)
    res = cross_validate(pipe, X_train, Y_train, cv=FOLDS, scoring=scoring, return_train_score=True)
    means = pd.DataFrame(res).mean().to_dict()
    # fit_time and score_time are wall-clock measurements, so only scores are comparable.
    return {k: v for k, v in means.items() if k.startswith(('test_', 'train_'))}


def test_get_or_create_experiment_reuses_existing_experiment(mlflow_client):
    name = unique_name('reuse')
    first = get_or_create_experiment(name)
    second = get_or_create_experiment(name)
    assert first == second
    matches = [e for e in mlflow_client.search_experiments() if e.name == name]
    assert len(matches) == 1


def test_run_cv_logs_params_and_exact_cv_means_without_model(credit_sample):
    X, Y = credit_sample
    name = unique_name('cv_no_model')
    scoring = ['neg_log_loss', 'accuracy']

    metrics = run_cv(
        get_pipe(), X, Y, PARAMS, folds=FOLDS, experiment_name=name, test_size=TEST_SIZE,
        scoring=scoring, random_state=SPLIT_SEED, tags={'purpose': 'test'}, log_model=False,
    )

    expected = _expected_cv_means(X, Y, scoring)
    assert metrics.keys() == expected.keys() | {'fit_time', 'score_time'}
    assert set(expected) == {'test_neg_log_loss', 'train_neg_log_loss', 'test_accuracy', 'train_accuracy'}
    for key, value in expected.items():
        assert metrics[key] == pytest.approx(value)

    (run,) = experiment_runs(name)
    assert run.data.params == {
        'clf__C': '0.5',
        'clf__random_state': '0',
        'folds': str(FOLDS),
        'test_size': str(TEST_SIZE),
        'random_state': str(SPLIT_SEED),
        'scoring': "['neg_log_loss', 'accuracy']",
    }
    for key, value in expected.items():
        assert run.data.metrics[key] == pytest.approx(value)
    assert run.data.tags['purpose'] == 'test'
    logged = mlflow.search_logged_models(
        experiment_ids=[run.info.experiment_id], output_format='list'
    )
    assert logged == []


def test_run_cv_with_model_registers_a_loadable_model(credit_sample):
    X, Y = credit_sample
    model_name = unique_name('CreditTestModel')

    run_cv(
        get_pipe(), X, Y, PARAMS, folds=2, experiment_name=unique_name('cv_model'),
        model_name=model_name, random_state=1, log_model=True,
    )

    model = mlflow.sklearn.load_model(f'models:/{model_name}/1')
    assert model is not None
    assert model.named_steps['clf'].C == 0.5
    assert_valid_probabilities(model, X)


def test_string_scoring_behaves_like_single_item_list(credit_sample):
    X, Y = credit_sample
    as_string = run_cv(
        get_pipe(), X, Y, PARAMS, folds=2, experiment_name=unique_name('cv_str'),
        scoring='neg_log_loss', random_state=1, log_model=False,
    )
    as_list = run_cv(
        get_pipe(), X, Y, PARAMS, folds=2, experiment_name=unique_name('cv_list'),
        scoring=['neg_log_loss'], random_state=1, log_model=False,
    )
    assert as_string.keys() == as_list.keys()
    assert as_string['test_neg_log_loss'] == pytest.approx(as_list['test_neg_log_loss'])


def test_nested_run_is_a_child_of_the_active_parent(credit_sample):
    X, Y = credit_sample
    name = unique_name('cv_nested')
    mlflow.set_experiment(experiment_id=get_or_create_experiment(name))

    with mlflow.start_run() as parent:
        run_cv(get_pipe(), X, Y, PARAMS, folds=2, random_state=1, nested=True, log_model=False)

    runs = experiment_runs(name)
    children = [r for r in runs if r.info.run_id != parent.info.run_id]
    assert len(runs) == 2
    assert len(children) == 1
    assert children[0].data.tags['mlflow.parentRunId'] == parent.info.run_id
