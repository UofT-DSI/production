"""
Pytest plugin that enforces exactly one layer marker per test.

Every test in this suite belongs to one layer:

  unit       — fast, exact-value checks of a single function or class
  pipeline   — end-to-end data pipeline runs into temporary directories
  experiment — MLflow-logging runs against the temporary tracking store

A test with no layer marker, or with more than one, aborts the run with a
usage error naming the offending tests.
"""

import pytest

LAYERS = ('unit', 'pipeline', 'experiment')


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    offenders = []
    for item in items:
        layers = [m.name for m in item.iter_markers() if m.name in LAYERS]
        if len(set(layers)) != 1:
            offenders.append(f'{item.nodeid} (layers: {sorted(set(layers)) or "none"})')
    if offenders:
        raise pytest.UsageError(
            'Every test needs exactly one layer marker '
            f'({", ".join(LAYERS)}):\n  ' + '\n  '.join(offenders)
        )
