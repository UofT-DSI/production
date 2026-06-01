"""
Logistic regression pipeline factory for credit risk experiments.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler


def get_pipe() -> Pipeline:
    """Build and return the sklearn preprocessing + logistic regression pipeline.

    Two preprocessing branches run in parallel via ColumnTransformer:

    ``num_standard`` — count and age columns:
        SimpleImputer (median) → StandardScaler

    ``num_pow_cols`` — skewed continuous columns (utilization, income, debt ratio):
        SimpleImputer (median) → StandardScaler → PowerTransformer (Yeo-Johnson)

    Binary indicator columns created in ``data.load_data``
    (``high_debt_ratio``, ``missing_monthly_income``, ``missing_num_dependents``)
    pass through the ColumnTransformer unchanged via ``remainder='passthrough'``.

    Returns
    -------
    Pipeline
        Unfitted pipeline ready for ``set_params`` and ``fit``.
    """
    num_std_cols = [
        'num_30_59_days_late',
        'num_60_89_days_late',
        'num_90_days_late',
        'num_open_credit_loans',
        'num_real_estate_loans',
        'age',
        'num_dependents',
    ]
    num_pow_cols = [
        'revolving_unsecured_line_utilization',
        'monthly_income',
        'debt_ratio',
    ]

    preproc_std = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    preproc_power = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('power', PowerTransformer()),
    ])

    ct = ColumnTransformer(transformers=[
        ('num_standard', preproc_std, num_std_cols),
        ('num_pow_cols', preproc_power, num_pow_cols),
    ], remainder='passthrough')

    return Pipeline(steps=[
        ('preproc', ct),
        ('clf', LogisticRegression()),
    ])
