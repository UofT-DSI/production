"""
Session isolation for the DSI production test suite.

Everything a test could leak is redirected into one session temp root
before any source module is imported:

  - the working directory is set to ``05_src`` so the relative paths in
    ``05_src/.env`` resolve the same way they do for the course scripts;
  - ``LOG_DIR`` points at ``<root>/logs`` so test runs never write to ``07_logs``;
  - MLflow tracking and the Model Registry use a SQLite database in ``<root>``,
    and every artifact lands under ``<root>/artifacts``.

``credit.experiment`` is imported here, first, because it runs
``load_dotenv(override=True)`` and ``mlflow.set_tracking_uri`` at import time;
the redirect has to happen after that, or the class server would win.
"""

import os
import tempfile
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1]
TESTS_README = '05_src/tests/readme.md'
SESSION_ROOT = Path(tempfile.mkdtemp(prefix='dsi_tests_'))

os.chdir(SRC_DIR)
os.environ['LOG_DIR'] = str(SESSION_ROOT / 'logs')
os.environ['MLFLOW_DISABLE_AGENT_HINT'] = '1'

import credit.experiment  # noqa: F401
import mlflow

# credit.experiment just ran load_dotenv(override=True), resetting LOG_DIR and the MLflow URI.
MLFLOW_URI = f'sqlite:///{(SESSION_ROOT / "mlflow.db").as_posix()}'
os.environ['LOG_DIR'] = str(SESSION_ROOT / 'logs')
os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_URI
os.environ['MLFLOW_REGISTRY_URI'] = MLFLOW_URI
os.environ['_MLFLOW_SERVER_ARTIFACT_ROOT'] = (SESSION_ROOT / 'artifacts').as_uri()
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_registry_uri(MLFLOW_URI)


def require_data_path(env_var: str, must_be_dir: bool = False) -> Path:
    """Return the path named by ``env_var``, or fail the test loudly.

    Missing data is a setup error, not a reason to skip: a partial green run
    would hide a broken environment.
    """
    value = os.getenv(env_var)
    if not value:
        pytest.fail(
            f'{env_var} is not set. Add it to 05_src/.env as described in {TESTS_README}.',
            pytrace=False,
        )
    path = Path(value).resolve()
    exists = path.is_dir() if must_be_dir else path.is_file()
    if not exists:
        kind = 'directory' if must_be_dir else 'file'
        pytest.fail(
            f'{env_var} points to a missing {kind}: {path}. '
            f'See {TESTS_README} for how to obtain the data.',
            pytrace=False,
        )
    return path
