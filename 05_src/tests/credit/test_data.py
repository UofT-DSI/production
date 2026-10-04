"""
Unit tests for ``credit.data.load_data``.
"""

import numpy as np
import pandas as pd
import pytest
from credit.data import load_data

pytestmark = pytest.mark.unit

FEATURE_COLUMNS = [
    'revolving_unsecured_line_utilization',
    'age',
    'num_30_59_days_late',
    'debt_ratio',
    'monthly_income',
    'num_open_credit_loans',
    'num_90_days_late',
    'num_real_estate_loans',
    'num_60_89_days_late',
    'num_dependents',
]

RAW_COLUMNS = [
    'Unnamed: 0',
    'SeriousDlqin2yrs',
    'RevolvingUtilizationOfUnsecuredLines',
    'age',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents',
]


def test_real_file_has_expected_shape_and_columns(credit_xy):
    X, Y = credit_xy
    assert X.shape == (150_000, 10)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(Y) == 150_000
    assert Y.name == 'delinquency'


def test_real_target_is_binary(credit_xy):
    _, Y = credit_xy
    assert set(Y.unique()) == {0, 1}


def test_real_features_are_numeric(credit_xy):
    X, _ = credit_xy
    assert all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)


def test_non_numeric_values_are_coerced_to_nan(tmp_path):
    rows = [
        [1, 0, 0.5, 40, 0, 0.3, '5000', 4, 0, 1, 0, 2],
        [2, 1, 0.9, 55, 2, 1.2, 'NA_TEXT', 7, 1, 0, 0, 'unknown'],
    ]
    path = tmp_path / 'credit.csv'
    pd.DataFrame(rows, columns=RAW_COLUMNS).to_csv(path, index=False)

    X, Y = load_data(str(path))

    assert X['monthly_income'].tolist()[0] == 5000
    assert np.isnan(X['monthly_income'].tolist()[1])
    assert np.isnan(X['num_dependents'].tolist()[1])
    assert Y.tolist() == [0, 1]
    assert 'Unnamed: 0' not in X.columns
