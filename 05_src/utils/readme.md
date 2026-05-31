# utils

Shared utilities for the DSI production project.

---

## `logger.py` — Logger factory

Returns a named `logging.Logger` pre-configured with two handlers:

| Handler | Destination | Format fields |
|---------|-------------|---------------|
| `FileHandler` | `<log_dir>/<YYYYMMDD_HHMMSS>.log` | asctime, name, filename, lineno, funcName, levelname, message |
| `StreamHandler` | stdout | asctime, filename, lineno, levelname, message |

Python's `logging` module is a global registry: the first call for a given `name` creates and caches the logger; subsequent calls return the same instance without adding duplicate handlers.

### `get_logger`

```python
get_logger(name, log_dir=LOG_DIR, log_level=LOG_LEVEL) -> logging.Logger
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | Logger name, typically `__name__` of the calling module. |
| `log_dir` | `str` | `LOG_DIR` env var, or `'./logs/'` | Directory for log files. Created automatically if absent. |
| `log_level` | `str` | `LOG_LEVEL` env var, or `'INFO'` | Minimum severity level (`'DEBUG'`, `'INFO'`, `'WARNING'`, …). |

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LOG_DIR` | `./logs/` | Root directory for all log files. |
| `LOG_LEVEL` | `INFO` | Minimum severity level written to both handlers. |

### Example

```python
from utils.logger import get_logger

log = get_logger(__name__)

log.info("pipeline started")
log.warning("missing optional config key: 'batch_size'")
log.error("failed to connect to database", exc_info=True)
```

Calling `get_logger(__name__)` from multiple modules in the same process is safe — each module gets its own named logger, and no handler is registered more than once per name.
