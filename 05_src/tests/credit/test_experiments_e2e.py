"""
End-to-end tests for the four credit experiment entry points.

Each experiment runs once, on the full real credit data, with the smallest
settings that still exercise its logic (2 folds, 1–3 trials). The tests then
read back what landed in the session's temporary MLflow store.
"""

import math

import mlflow
import mlflow.sklearn
import optuna
import pytest
from credit.exp__linear_hyperopt import linear_search
from credit.exp__logistic_grid_search import grid_search
from credit.exp__logistic_hyperopt import hyperparam_opt
from credit.exp__logistic_simple import single_run
from sklearn.model_selection import ParameterGrid
from tests.harness import SRC_DIR
from tests.helpers import assert_valid_probabilities, experiment_runs, unique_name

pytestmark = pytest.mark.experiment


def _split_parent_children(runs: list) -> tuple:
    parents = [r for r in runs if 'mlflow.parentRunId' not in r.data.tags]
    children = [r for r in runs if 'mlflow.parentRunId' in r.data.tags]
    assert len(parents) == 1
    assert all(c.data.tags['mlflow.parentRunId'] == parents[0].info.run_id for c in children)
    return parents[0], children


def _load_latest(model_name: str):
    model = mlflow.sklearn.load_model(f'models:/{model_name}/latest')
    assert model is not None
    return model


@pytest.fixture(scope='module')
def simple_runs(credit_csv):
    single_run(folds=2)
    return experiment_runs('credit_single_run_logistic')


@pytest.fixture(scope='module')
def grid_runs(credit_csv):
    grid_search(folds=2)
    return experiment_runs('credit_grid_search_logistic')


@pytest.fixture(scope='module')
def hyperopt_result(credit_csv):
    experiment_name, model_name = unique_name('hyperopt'), unique_name('CreditLogisticHyperopt')
    hyperparam_opt(folds=2, max_evals=2, experiment_name=experiment_name, model_name=model_name)
    return experiment_runs(experiment_name), model_name


@pytest.fixture(scope='module')
def linear_result(credit_csv):
    experiment_name, model_name = unique_name('linear'), unique_name('CreditLinear')
    linear_search(folds=2, n_trials=3, experiment_name=experiment_name, model_name=model_name)
    return experiment_runs(experiment_name), model_name


def test_single_run_logs_one_run_with_finite_metrics(simple_runs):
    assert len(simple_runs) == 1
    run = simple_runs[0]
    assert run.data.tags['optimizer'] == 'none'
    metrics = run.data.metrics
    assert all(math.isfinite(metrics[k]) for k in ('test_neg_log_loss', 'test_accuracy', 'test_f1'))
    assert metrics['test_neg_log_loss'] < 0
    assert 0 <= metrics['test_accuracy'] <= 1
    assert 0 <= metrics['test_f1'] <= 1


def test_single_run_registers_a_loadable_model(simple_runs, credit_xy):
    model = _load_latest('CreditLogisticSimple')
    assert model.named_steps['clf'].C == 1.0
    assert_valid_probabilities(model, credit_xy[0])


def test_grid_search_logs_one_run_per_grid_point(grid_runs):
    space = {
        'preproc__num_standard__imputer__add_indicator': ['False', 'True'],
        'clf__C': ['0.1', '0.25', '0.5', '0.75', '0.9', '1.0'],
        'clf__class_weight': ['None', 'balanced'],
    }
    expected = sorted(tuple(sorted(p.items())) for p in ParameterGrid(space))
    logged = sorted(tuple(sorted((k, r.data.params[k]) for k in space)) for r in grid_runs)
    assert len(grid_runs) == 24
    assert logged == expected
    assert {r.data.tags['optimizer'] for r in grid_runs} == {'grid_search'}


def test_grid_search_registers_no_model(grid_runs):
    experiment_ids = {r.info.experiment_id for r in grid_runs}
    assert mlflow.search_logged_models(experiment_ids=list(experiment_ids), output_format='list') == []


def test_hyperopt_logs_parent_with_best_child_score(hyperopt_result):
    runs, _ = hyperopt_result
    parent, children = _split_parent_children(runs)
    assert len(children) == 2
    best_child = max(c.data.metrics['test_neg_log_loss'] for c in children)
    assert parent.data.metrics['best_test_neg_log_loss'] == pytest.approx(best_child)


def test_hyperopt_registers_a_loadable_model(hyperopt_result, credit_xy):
    _, model_name = hyperopt_result
    assert_valid_probabilities(_load_latest(model_name), credit_xy[0])


def test_linear_search_logs_parent_and_one_child_per_trial(linear_result):
    runs, _ = linear_result
    _, children = _split_parent_children(runs)
    assert len(children) == 3


def test_linear_search_refits_with_the_best_trials_full_params(linear_result, credit_xy):
    runs, model_name = linear_result
    _, children = _split_parent_children(runs)
    best = max(children, key=lambda c: c.data.metrics['test_neg_log_loss']).data.params

    model = _load_latest(model_name)
    clf = model.named_steps['clf']
    assert str(clf.C) == best['clf__C']
    assert clf.solver == best['clf__solver']
    assert str(clf.l1_ratio) == best['clf__l1_ratio']
    assert str(clf.class_weight) == best['clf__class_weight']
    assert_valid_probabilities(model, credit_xy[0])


def test_linear_search_with_l1_best_trial_registers_saga_l1_model(credit_csv, monkeypatch):
    create_study = optuna.create_study

    def create_study_with_l1_first(*args, **kwargs):
        study = create_study(*args, **kwargs)
        study.enqueue_trial({'clf__penalty': 'l1', 'clf__C': 1.0, 'clf__class_weight': None})
        return study

    monkeypatch.setattr(optuna, 'create_study', create_study_with_l1_first)
    model_name = unique_name('CreditLinearL1')
    linear_search(folds=2, n_trials=1, experiment_name=unique_name('linear_l1'), model_name=model_name)

    clf = _load_latest(model_name).named_steps['clf']
    assert clf.solver == 'saga'
    assert clf.l1_ratio == 1.0


def test_no_mlflow_files_are_written_under_05_src():
    assert not (SRC_DIR / 'mlruns').exists()
    assert not (SRC_DIR / 'mlartifacts').exists()
    assert not (SRC_DIR / 'mlflow.db').exists()
