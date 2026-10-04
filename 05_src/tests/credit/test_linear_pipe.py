"""
Unit tests for the ``credit.linear`` feature-engineering pipeline.

Each branch is checked on a small hand-built frame. ``branch[:-1]`` drops the
final ``StandardScaler`` so the clip / log / impute values can be compared
exactly before scaling.
"""

import numpy as np
import pandas as pd
import pytest
from credit.linear import get_pipe

pytestmark = pytest.mark.unit


def _branch(name: str):
    ct = get_pipe().named_steps['preproc']
    return {n: t for n, t, _ in ct.transformers}[name]


def _unscaled(name: str, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(_branch(name)[:-1].fit_transform(frame), dtype=float)


def _full_row_frame() -> pd.DataFrame:
    return pd.DataFrame({
        'num_30_59_days_late': [0, 98, 1],
        'num_60_89_days_late': [0, 96, 0],
        'num_90_days_late': [0, 98, 2],
        'revolving_unsecured_line_utilization': [0.1, 3.0, 0.5],
        'debt_ratio': [0.2, 10.0, 1.0],
        'monthly_income': [5000.0, np.nan, 3000.0],
        'age': [30, 50, 120],
        'num_open_credit_loans': [3, 40, 5],
        'num_real_estate_loans': [1, 12, 0],
        'num_dependents': [np.nan, 20, 2],
    })


def test_late_payment_counts_are_clipped_to_15():
    frame = pd.DataFrame({
        'num_30_59_days_late': [0, 3, 96, 98],
        'num_60_89_days_late': [0, 3, 96, 98],
        'num_90_days_late': [0, 3, 96, 98],
    })
    out = _unscaled('late', frame)
    assert out[:, 0] == pytest.approx([0, 3, 15, 15])


def test_utilization_is_clipped_to_0_2_then_log1p():
    out = _unscaled('util', pd.DataFrame({'revolving_unsecured_line_utilization': [-1.0, 0.5, 5.0]}))
    assert out[:, 0] == pytest.approx(np.log1p([0.0, 0.5, 2.0]))


def test_debt_ratio_is_clipped_to_5_then_log1p():
    out = _unscaled('debt', pd.DataFrame({'debt_ratio': [0.0, 1.0, 10.0]}))
    assert out[:, 0] == pytest.approx(np.log1p([0.0, 1.0, 5.0]))


def test_income_nan_is_imputed_with_median_then_log1p():
    out = _unscaled('income', pd.DataFrame({'monthly_income': [1000.0, np.nan, 3000.0]}))
    assert out[:, 0] == pytest.approx(np.log1p([1000.0, 2000.0, 3000.0]))


def test_age_is_clipped_to_18_105():
    out = _unscaled('age', pd.DataFrame({'age': [10, 50, 120]}))
    assert out[:, 0] == pytest.approx([18, 50, 105])


def test_open_loans_and_real_estate_are_clipped():
    assert _unscaled('open_loans', pd.DataFrame({'num_open_credit_loans': [5, 40]}))[:, 0] == pytest.approx([5, 30])
    assert _unscaled('real_estate', pd.DataFrame({'num_real_estate_loans': [1, 12]}))[:, 0] == pytest.approx([1, 10])


def test_dependents_are_clipped_then_imputed():
    out = _unscaled('dep', pd.DataFrame({'num_dependents': [np.nan, 20.0]}))
    # 20 is clipped to 10, and the NaN takes the median of the clipped values (10).
    assert out[:, 0] == pytest.approx([10, 10])


def test_missing_indicators_are_kept_unscaled_in_full_output():
    ct = get_pipe().named_steps['preproc']
    out = ct.fit_transform(_full_row_frame())
    income_ind = out[:, ct.output_indices_['income_ind']].ravel()
    dep_ind = out[:, ct.output_indices_['dep_ind']].ravel()
    assert income_ind.tolist() == [0.0, 1.0, 0.0]
    assert dep_ind.tolist() == [1.0, 0.0, 0.0]


def test_full_output_has_twelve_columns_and_no_nan():
    ct = get_pipe().named_steps['preproc']
    out = ct.fit_transform(_full_row_frame())
    assert out.shape == (3, 12)
    assert not np.isnan(out).any()


def test_fitted_pipeline_returns_valid_probabilities(credit_sample):
    X, Y = credit_sample
    pipe = get_pipe().fit(X, Y)
    proba = pipe.predict_proba(X.head(50))
    assert proba.shape == (50, 2)
    assert proba.sum(axis=1) == pytest.approx(np.ones(50))
    assert ((proba >= 0) & (proba <= 1)).all()
