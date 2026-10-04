"""
Unit tests for ``credit.exp__linear_hyperopt.suggest_params``.

``optuna.trial.FixedTrial`` makes each branch of the search space deterministic.
"""

import optuna
import pytest
from credit.exp__linear_hyperopt import suggest_params

pytestmark = pytest.mark.unit


def _params(**fixed) -> dict:
    base = {'clf__C': 0.5, 'clf__class_weight': None}
    return suggest_params(optuna.trial.FixedTrial({**base, **fixed}), random_state=7)


def test_l1_uses_saga_and_l1_ratio_one():
    params = _params(clf__penalty='l1')
    assert params['clf__solver'] == 'saga'
    assert params['clf__l1_ratio'] == 1.0


def test_elasticnet_uses_saga_and_sampled_l1_ratio():
    params = _params(clf__penalty='elasticnet', clf__l1_ratio=0.3)
    assert params['clf__solver'] == 'saga'
    assert params['clf__l1_ratio'] == pytest.approx(0.3)


@pytest.mark.parametrize('solver', ['lbfgs', 'saga'])
def test_l2_uses_sampled_solver_and_l1_ratio_zero(solver):
    params = _params(clf__penalty='l2', clf__solver=solver)
    assert params['clf__solver'] == solver
    assert params['clf__l1_ratio'] == 0.0


def test_shared_params_are_passed_through():
    params = _params(clf__penalty='l2', clf__solver='lbfgs', clf__C=2.5, clf__class_weight='balanced')
    assert params == {
        'clf__C': 2.5,
        'clf__solver': 'lbfgs',
        'clf__class_weight': 'balanced',
        'clf__l1_ratio': 0.0,
        'clf__random_state': 7,
    }
