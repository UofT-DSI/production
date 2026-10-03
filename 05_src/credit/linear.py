"""
Logistic regression pipeline with expanded feature engineering for credit risk experiments.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def _clip(lo: float, hi: float) -> FunctionTransformer:
    return FunctionTransformer(np.clip, kw_args={'a_min': lo, 'a_max': hi})


def get_pipe() -> Pipeline:
    """Build the preprocessing + logistic regression pipeline.

    Per-variable transformations applied before scaling:

    | Branch | Columns | Steps |
    |--------|---------|-------|
    | ``late`` | num_*_days_late (×3) | clip [0, 15] → impute median |
    | ``util`` | revolving_unsecured_line_utilization | clip [0, 2] → log1p |
    | ``debt`` | debt_ratio | clip [0, 5] → log1p |
    | ``income`` | monthly_income | impute median → log1p |
    | ``income_ind`` | monthly_income | MissingIndicator |
    | ``age`` | age | clip [18, 105] |
    | ``open_loans`` | num_open_credit_loans | clip [0, 30] |
    | ``real_estate`` | num_real_estate_loans | clip [0, 10] |
    | ``dep`` | num_dependents | clip [0, 10] → impute median |
    | ``dep_ind`` | num_dependents | MissingIndicator |

    Clipping thresholds for late-payment counts address the known data-quality
    issue where values of 96 and 98 appear to be coding errors for "never".
    Utilization and debt ratio are log-compressed after capping to reduce the
    influence of extreme outliers on the linear decision boundary.
    Missing indicators for income and dependents are kept as raw binary columns
    (not scaled) so the missingness signal is preserved.

    Returns
    -------
    Pipeline
        Unfitted pipeline ready for ``set_params`` and ``fit``.
    """
    late_cols = ['num_30_59_days_late', 'num_60_89_days_late', 'num_90_days_late']

    preproc_late = Pipeline([
        ('clip', _clip(0, 15)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    preproc_util = Pipeline([
        ('clip', _clip(0, 2)),
        ('log1p', FunctionTransformer(np.log1p)),
        ('scaler', StandardScaler()),
    ])

    preproc_debt = Pipeline([
        ('clip', _clip(0, 5)),
        ('log1p', FunctionTransformer(np.log1p)),
        ('scaler', StandardScaler()),
    ])

    preproc_income = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log1p', FunctionTransformer(np.log1p)),
        ('scaler', StandardScaler()),
    ])

    preproc_age = Pipeline([
        ('clip', _clip(18, 105)),
        ('scaler', StandardScaler()),
    ])

    preproc_open_loans = Pipeline([
        ('clip', _clip(0, 30)),
        ('scaler', StandardScaler()),
    ])

    preproc_real_estate = Pipeline([
        ('clip', _clip(0, 10)),
        ('scaler', StandardScaler()),
    ])

    preproc_dep = Pipeline([
        ('clip', _clip(0, 10)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    ct = ColumnTransformer([
        ('late', preproc_late, late_cols),
        ('util', preproc_util, ['revolving_unsecured_line_utilization']),
        ('debt', preproc_debt, ['debt_ratio']),
        ('income', preproc_income, ['monthly_income']),
        ('income_ind', MissingIndicator(features='all'), ['monthly_income']),
        ('age', preproc_age, ['age']),
        ('open_loans', preproc_open_loans, ['num_open_credit_loans']),
        ('real_estate', preproc_real_estate, ['num_real_estate_loans']),
        ('dep', preproc_dep, ['num_dependents']),
        ('dep_ind', MissingIndicator(features='all'), ['num_dependents']),
    ])

    return Pipeline([
        ('preproc', ct),
        ('clf', LogisticRegression(max_iter=1000, solver='saga')),
    ])
