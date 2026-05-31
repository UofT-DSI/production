"""
Logger factory for the DSI production project.

Reads two environment variables (with defaults):
  LOG_DIR   — directory where log files are written  (default: './logs/')
  LOG_LEVEL — minimum severity level                 (default: 'INFO')

File log format  : asctime, name, filename, lineno, funcName, levelname, message
Stream log format: asctime, filename, lineno, levelname, message
"""

import logging
import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

LOG_DIR = os.getenv('LOG_DIR', './logs/')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

def get_logger(name: str, log_dir: str = LOG_DIR, log_level: str = LOG_LEVEL) -> logging.Logger:
    """Return a named logger with file and stream handlers.

    On the first call for a given `name` the function creates `log_dir` if it
    does not exist, attaches a timestamped ``FileHandler`` and a
    ``StreamHandler``, then sets the level.  Subsequent calls with the same
    `name` return the cached logger unchanged (Python's ``logging`` module
    is a global registry).

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__`` of the calling module.
    log_dir : str
        Directory for log files.  Created automatically if absent.
        Defaults to the ``LOG_DIR`` env var, or ``'./logs/'``.
    log_level : str
        Minimum severity level (e.g. ``'DEBUG'``, ``'INFO'``, ``'WARNING'``).
        Defaults to the ``LOG_LEVEL`` env var, or ``'INFO'``.

    Returns
    -------
    logging.Logger

    Example
    -------
    >>> from utils.logger import get_logger
    >>> log = get_logger(__name__)
    >>> log.info("pipeline started")
    """
    _logs = logging.getLogger(name)

    if not len(_logs.handlers):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        f_handler = logging.FileHandler(os.path.join(log_dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))
        f_format = logging.Formatter('%(asctime)s, %(name)s, %(filename)s, %(lineno)d, %(funcName)s, %(levelname)s, %(message)s')
        f_handler.setFormatter(f_format)

        s_handler = logging.StreamHandler()
        s_format = logging.Formatter('%(asctime)s, %(filename)s, %(lineno)d, %(levelname)s, %(message)s')
        s_handler.setFormatter(s_format)

        _logs.addHandler(f_handler)
        _logs.addHandler(s_handler)

    _logs.setLevel(log_level)
    return _logs