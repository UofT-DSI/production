import pandas as pd
from sacred import Ingredient
import os
from dotenv import load_dotenv
from logger import get_logger

_logs = get_logger(__name__)

data_ingredient = Ingredient('data_ingredient')

data_ingredient.logger = _logs

@data_ingredient.config
def cfg():
    ft_file = os.getenv("CREDIT_DATA")

@data_ingredient.capture
def get_data(ft_file = os.getenv("CREDIT_DATA")):
    '''Loads data from a given location.'''
    _logs.info(f'Getting data from {ft_file}')
    df_raw = pd.read_csv(ft_file)
    return df_raw

@data_ingredient.capture
def process_data(df_raw):
    '''Update data names, add features, and remove unwanted columns.'''
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
    )
    return df

@data_ingredient.capture
def load_data():
    '''Load data and return X, Y'''
    df_raw = get_data()
    df = process_data(df_raw)
    X = df.drop(columns = ['delinquency'])
    Y = df['delinquency']
    return X, Y
