"""
Unit tests for the ``credit.logistic`` preprocessing + logistic regression pipeline.
"""

import numpy as np
import pandas as pd
import pytest
from credit.logistic import get_pipe

pytestmark = pytest.mark.unit

STANDARD_COLUMNS = [
    'num_30_59_days_late',
    'num_60_89_days_late',
    'num_90_days_late',
    'num_open_credit_loans',
    'num_real_estate_loans',
    'age',
    'num_dependents',
]
POWER_COLUMNS = ['revolving_unsecured_line_utilization', 'monthly_income', 'debt_ratio']


def test_preprocessor_routes_documented_columns_to_each_branch():
    ct = get_pipe().named_steps['preproc']
    routing = {name: cols for name, _, cols in ct.transformers}
    assert routing == {'num_standard': STANDARD_COLUMNS, 'num_pow_cols': POWER_COLUMNS}


def test_preprocessor_outputs_one_column_per_feature(credit_sample):
    X, _ = credit_sample
    ct = get_pipe().named_steps['preproc']
    out = ct.fit_transform(X)
    assert out.shape == (len(X), 10)
    assert not np.isnan(out).any()


def test_standard_branch_imputes_median_then_standardises():
    ct = get_pipe().named_steps['preproc']
    branch = {name: t for name, t, _ in ct.transformers}['num_standard']
    frame = pd.DataFrame({c: [1.0, 2.0, 3.0, np.nan] for c in STANDARD_COLUMNS})
    out = branch.fit_transform(frame)
    # NaN is imputed with the median (2.0); the column [1, 2, 3, 2] is then standardised.
    expected = (np.array([1, 2, 3, 2]) - 2.0) / np.std([1, 2, 3, 2])
    assert out[:, 0] == pytest.approx(expected)


def test_fitted_pipeline_returns_valid_probabilities(credit_sample):
    X, Y = credit_sample
    pipe = get_pipe().fit(X, Y)
    proba = pipe.predict_proba(X.head(50))
    assert proba.shape == (50, 2)
    assert proba.sum(axis=1) == pytest.approx(np.ones(50))
    assert ((proba >= 0) & (proba <= 1)).all()
