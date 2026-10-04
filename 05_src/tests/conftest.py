"""
Shared fixtures for the DSI production test suite.

Importing ``tests.harness`` first isolates logs and MLflow for the whole
session; see that module for details. The ``_run_from_05_src`` fixture then
makes ``05_src`` the working directory so the relative paths in ``05_src/.env``
resolve the same way they do for the course scripts. It is a fixture, not an
import-time ``chdir``, because pytest resolves ``testpaths`` after conftest
import: an early ``chdir`` makes it collect all of ``05_src``. Data-dependent
tests request ``credit_csv`` / ``price_csv_dir``, which fail (never skip)
when the real data is missing.
"""

import logging
import shutil
from pathlib import Path

import mlflow
import pytest
from tests.harness import SESSION_ROOT, SRC_DIR, TESTS_README, require_data_path
from tests.layer_markers import pytest_collection_modifyitems  # noqa: F401


@pytest.fixture(scope='session', autouse=True)
def _run_from_05_src():
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(SRC_DIR)
        yield


@pytest.fixture(scope='session')
def credit_csv() -> Path:
    """Path to the real Give Me Some Credit training CSV."""
    return require_data_path('CREDIT_DATA')


@pytest.fixture(scope='session')
def price_csv_dir() -> Path:
    """Directory holding one real price CSV per ticker (``<PRICE_CSV_DATA>/stocks``)."""
    stocks = require_data_path('PRICE_CSV_DATA', must_be_dir=True) / 'stocks'
    if not stocks.is_dir():
        pytest.fail(
            f'Expected per-ticker CSVs in {stocks}. See {TESTS_README}.',
            pytrace=False,
        )
    return stocks


@pytest.fixture(scope='session')
def credit_xy(credit_csv: Path):
    """The real credit data as ``(X, Y)``, loaded once per session."""
    from credit.data import load_data
    return load_data(str(credit_csv))


@pytest.fixture(scope='session')
def credit_sample(credit_xy):
    """A fixed-seed 5,000-row sample of the real credit data, for fast fits."""
    X, Y = credit_xy
    X_s = X.sample(n=5000, random_state=0)
    return X_s, Y.loc[X_s.index]


@pytest.fixture(scope='session')
def session_root() -> Path:
    return SESSION_ROOT


@pytest.fixture
def mlflow_client() -> mlflow.MlflowClient:
    return mlflow.MlflowClient()


@pytest.fixture(autouse=True)
def _close_leaked_mlflow_runs():
    yield
    while mlflow.active_run() is not None:
        mlflow.end_run()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    logging.shutdown()
    client = mlflow.MlflowClient()
    for store in (client._tracking_client.store, client._get_registry_client().store):
        engine = getattr(store, 'engine', None)
        if engine is not None:
            engine.dispose()
    shutil.rmtree(SESSION_ROOT, ignore_errors=True)
