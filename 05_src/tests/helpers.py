"""
Small helpers shared by several test modules.
"""

import uuid

import mlflow
import numpy as np
import pytest


def unique_name(prefix: str) -> str:
    """Return ``<prefix>_<random suffix>``.

    All tests in a session share one MLflow store, so experiment and model
    names must be unique or one test would read another test's runs.
    """
    return f'{prefix}_{uuid.uuid4().hex[:8]}'


def experiment_runs(experiment_name: str) -> list:
    """Return every run in the named experiment, failing if it does not exist."""
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    assert experiment is not None, f'experiment {experiment_name!r} was not created'
    return client.search_runs([experiment.experiment_id], max_results=1000)


def assert_valid_probabilities(model, X, n: int = 50) -> None:
    """Check ``model.predict_proba`` on the first ``n`` rows of ``X``."""
    proba = model.predict_proba(X.head(n))
    assert proba.shape == (n, 2)
    assert proba.sum(axis=1) == pytest.approx(np.ones(n))
    assert ((proba >= 0) & (proba <= 1)).all()
