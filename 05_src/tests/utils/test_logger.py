"""
Unit tests for ``utils.logger.get_logger``.

The ``logging`` module keeps a process-wide registry of loggers, so every test
uses its own logger name to stay independent of the others.
"""

import logging
import sys
import uuid

import pytest
from utils.logger import get_logger

pytestmark = pytest.mark.unit


@pytest.fixture
def name() -> str:
    return f'test_logger_{uuid.uuid4().hex}'


@pytest.fixture
def log_dir(tmp_path):
    return tmp_path / 'not_created_yet'


def _handlers_of_type(logger: logging.Logger, handler_type: type) -> list:
    return [h for h in logger.handlers if type(h) is handler_type]


def test_missing_log_dir_is_created(name, log_dir):
    assert not log_dir.exists()
    get_logger(name, log_dir=str(log_dir))
    assert log_dir.is_dir()


def test_first_call_attaches_one_file_and_one_stream_handler(name, log_dir):
    logger = get_logger(name, log_dir=str(log_dir))
    assert len(logger.handlers) == 2
    assert len(_handlers_of_type(logger, logging.FileHandler)) == 1
    assert len(_handlers_of_type(logger, logging.StreamHandler)) == 1


def test_repeat_call_returns_same_logger_without_duplicate_handlers(name, log_dir):
    first = get_logger(name, log_dir=str(log_dir))
    second = get_logger(name, log_dir=str(log_dir))
    assert second is first
    assert len(second.handlers) == 2


def test_log_level_is_applied_and_updated_on_later_calls(name, log_dir):
    logger = get_logger(name, log_dir=str(log_dir), log_level='DEBUG')
    assert logger.level == logging.DEBUG
    get_logger(name, log_dir=str(log_dir), log_level='WARNING')
    assert logger.level == logging.WARNING


def test_file_record_has_documented_fields_in_order(name, log_dir):
    logger = get_logger(name, log_dir=str(log_dir), log_level='INFO')
    logger.info('pipeline started')
    for handler in logger.handlers:
        handler.flush()

    log_files = list(log_dir.glob('*.log'))
    assert len(log_files) == 1
    lines = log_files[0].read_text().splitlines()
    assert len(lines) == 1

    # asctime carries its own comma before the milliseconds, so split on ", ".
    asctime, logger_name, filename, lineno, func_name, level, message = lines[0].split(', ')
    assert logger_name == name
    assert filename == 'test_logger.py'
    assert lineno.isdigit()
    assert func_name == 'test_file_record_has_documented_fields_in_order'
    assert level == 'INFO'
    assert message == 'pipeline started'
    assert asctime[:4].isdigit()


def test_records_below_level_are_not_written(name, log_dir):
    logger = get_logger(name, log_dir=str(log_dir), log_level='WARNING')
    logger.info('dropped')
    logger.warning('kept')
    for handler in logger.handlers:
        handler.flush()
    lines = next(log_dir.glob('*.log')).read_text().splitlines()
    assert [line.rsplit(', ', 1)[1] for line in lines] == ['kept']


def test_stream_handler_writes_to_stderr(name, log_dir):
    logger = get_logger(name, log_dir=str(log_dir))
    (stream_handler,) = _handlers_of_type(logger, logging.StreamHandler)
    assert stream_handler.stream is sys.stderr
