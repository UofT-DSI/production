"""
Data loading and preprocessing for the credit risk dataset.

Reads the raw Give Me Some Credit CSV, renames columns to snake_case,
engineers three binary missingness/threshold indicators, and returns
feature matrix X and target vector Y.

Reads one environment variable:
  CREDIT_DATA — path to the raw CSV file
"""

import os

import pandas as pd
from dotenv import load_dotenv

from utils.logger import get_logger

load_dotenv()
CREDIT_FILE = os.getenv("CREDIT_DATA")

_logs = get_logger(__name__)


def load_data(file: str | None = CREDIT_FILE) -> tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess the credit risk dataset.

    Performs the following steps in order:
    1. Drop the unnamed index column added by the original CSV export.
    2. Rename all columns to snake_case.
    3. Engineer three binary indicators:
       - ``high_debt_ratio``         — ``debt_ratio > 1``
       - ``missing_monthly_income``  — ``monthly_income`` is NaN
       - ``missing_num_dependents``  — ``num_dependents`` is NaN
    4. Coerce all columns to numeric, converting non-parseable values to NaN.
    5. Split into feature matrix X and target vector Y.

    Parameters
    ----------
    file : str | None
        Path to the raw CSV file. Defaults to the ``CREDIT_DATA`` env var.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with all columns except ``delinquency``.
    Y : pd.Series
        Binary target: 1 = serious delinquency within 2 years, 0 = no.
    """
    _logs.info(f'Loading credit data from {file}')
    df_raw = pd.read_csv(file)
    df = df_raw.drop(columns=['Unnamed: 0']).rename(
        columns={
            'SeriousDlqin2yrs': 'delinquency',
            'RevolvingUtilizationOfUnsecuredLines': 'revolving_unsecured_line_utilization',
            'age': 'age',
            'NumberOfTime30-59DaysPastDueNotWorse': 'num_30_59_days_late',
            'DebtRatio': 'debt_ratio',
            'MonthlyIncome': 'monthly_income',
            'NumberOfOpenCreditLinesAndLoans': 'num_open_credit_loans',
            'NumberOfTimes90DaysLate': 'num_90_days_late',
            'NumberRealEstateLoansOrLines': 'num_real_estate_loans',
            'NumberOfTime60-89DaysPastDueNotWorse': 'num_60_89_days_late',
            'NumberOfDependents': 'num_dependents',
        }
    ).assign(
        high_debt_ratio=lambda x: (x['debt_ratio'] > 1) * 1,
        missing_monthly_income=lambda x: x['monthly_income'].isna() * 1,
        missing_num_dependents=lambda x: x['num_dependents'].isna() * 1,
    ).apply(
        lambda x: pd.to_numeric(x, errors='coerce')
    )
    _logs.debug(f'Loaded {len(df)} rows, {df.shape[1]} columns')
    X = df.drop(columns=['delinquency'])
    Y = df['delinquency']
    return X, Y
