"""
Tests for the test harness itself: import isolation, MLflow redirection,
log redirection, loud data failures, and the one-layer-marker rule.
"""

import os
from pathlib import Path

import mlflow
import pytest
import utils.logger
from tests import layer_markers
from tests.harness import SESSION_ROOT, SRC_DIR, TESTS_README, require_data_path

pytestmark = pytest.mark.unit


def test_local_utils_package_wins_over_pypi_utils():
    assert Path(utils.logger.__file__).resolve().parent == SRC_DIR / 'utils'


def test_working_directory_is_05_src():
    assert Path.cwd().resolve() == SRC_DIR


def test_mlflow_tracking_and_registry_point_at_session_store():
    expected = f'sqlite:///{(SESSION_ROOT / "mlflow.db").as_posix()}'
    assert mlflow.get_tracking_uri() == expected
    assert mlflow.get_registry_uri() == expected
    assert 'localhost' not in mlflow.get_tracking_uri()


def test_log_dir_stays_in_session_root_after_credit_experiment_import():
    assert Path(os.environ['LOG_DIR']) == SESSION_ROOT / 'logs'


def test_missing_data_file_fails_loudly_naming_variable_and_readme(monkeypatch, tmp_path):
    monkeypatch.setenv('CREDIT_DATA', str(tmp_path / 'nope.csv'))
    with pytest.raises(pytest.fail.Exception) as excinfo:
        require_data_path('CREDIT_DATA')
    message = str(excinfo.value)
    assert 'CREDIT_DATA' in message
    assert TESTS_README in message


def test_unset_data_variable_fails_loudly(monkeypatch):
    monkeypatch.delenv('PRICE_CSV_DATA', raising=False)
    with pytest.raises(pytest.fail.Exception, match='PRICE_CSV_DATA is not set'):
        require_data_path('PRICE_CSV_DATA', must_be_dir=True)


def test_existing_data_file_is_returned_resolved(monkeypatch, tmp_path):
    data = tmp_path / 'data.csv'
    data.write_text('a\n1\n')
    monkeypatch.setenv('CREDIT_DATA', str(data))
    assert require_data_path('CREDIT_DATA') == data.resolve()


@pytest.fixture
def marker_pytester(pytester: pytest.Pytester) -> pytest.Pytester:
    pytester.makeini(
        '[pytest]\n'
        'markers =\n'
        '    unit: u\n'
        '    pipeline: p\n'
        '    experiment: e\n'
    )
    return pytester


def test_test_without_layer_marker_aborts_collection(marker_pytester):
    marker_pytester.makepyfile('def test_unmarked():\n    pass\n')
    result = marker_pytester.runpytest(plugins=[layer_markers])
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(['*test_unmarked*layers: none*'])


def test_test_with_two_layer_markers_aborts_collection(marker_pytester):
    marker_pytester.makepyfile(
        'import pytest\n'
        '@pytest.mark.unit\n'
        '@pytest.mark.pipeline\n'
        'def test_double():\n'
        '    pass\n'
    )
    result = marker_pytester.runpytest(plugins=[layer_markers])
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*test_double*['pipeline', 'unit']*"])


def test_test_with_one_layer_marker_runs(marker_pytester):
    marker_pytester.makepyfile(
        'import pytest\n'
        '@pytest.mark.experiment\n'
        'def test_single():\n'
        '    pass\n'
    )
    result = marker_pytester.runpytest(plugins=[layer_markers])
    result.assert_outcomes(passed=1)
