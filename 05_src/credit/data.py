
import pandas as pd
import os

from dotenv import load_dotenv
from utils.logger import get_logger

load_dotenv()
CREDIT_FILE = os.getenv("CREDIT_DATA")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")

_logs = get_logger(__name__)

def load_data(file = CREDIT_FILE):
    '''Loads data from a given location.'''
    _logs.info(f'Getting data from {file}')
    df_raw = pd.read_csv(file)
    df = df_raw.drop(columns = ["Unnamed: 0"]).rename(
        columns = {
            'SeriousDlqin2yrs': 'delinquency',
            'RevolvingUtilizationOfUnsecuredLines': 'revolving_unsecured_line_utilization', 
            'age': 'age',
            'NumberOfTime30-59DaysPastDueNotWorse': 'num_30_59_days_late', 
            'DebtRatio': 'debt_ratio', 
            'MonthlyIncome': 'monthly_income',
            'NumberOfOpenCreditLinesAndLoans': 'num_open_credit_loans', 
            'NumberOfTimes90DaysLate':  'num_90_days_late',
            'NumberRealEstateLoansOrLines': 'num_real_estate_loans', 
            'NumberOfTime60-89DaysPastDueNotWorse': 'num_60_89_days_late',
            'NumberOfDependents': 'num_dependents'
        }
    ).assign(
        high_debt_ratio = lambda x: (x['debt_ratio'] > 1)*1,
        missing_monthly_income = lambda x: x['monthly_income'].isna()*1,
        missing_num_dependents = lambda x: x['num_dependents'].isna()*1, 
    ).apply(
        lambda x: pd.to_numeric(x, errors='coerce', downcast='float')
    )
    _logs.debug(f'Data info: {df.info()}')
    X = df.drop(columns = ['delinquency'])
    Y = df['delinquency']
    
    return X, Y