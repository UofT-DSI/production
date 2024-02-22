import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sacred import Ingredient
from logger import get_logger

_logs = get_logger(__name__)

preproc_ingredient = Ingredient('preproc_ingredient')

preproc_ingredient.logger = _logs

@preproc_ingredient.config
def cfg():
    '''
    Config function of the ingredient: all values defined here are shared across captured functions.
    '''
    num_not_transform = ['num_30_59_days_late', 
                     'num_60_89_days_late', 
                     'num_90_days_late',
                     'num_open_credit_loans', 
                     'num_real_estate_loans',
                     'age', 'num_dependents']
    num_transform = ['revolving_unsecured_line_utilization', 'monthly_income', 'debt_ratio']
    remainder = "passthrough"


@preproc_ingredient.capture
def get_column_transformer(preproc_pipe, num_not_transform, num_transform, remainder):
    '''
    Get a column transformer given a name of the preproc_pipe and the column names.
    '''
    _logs.info(f'Getting Column Transformer {preproc_pipe}')
    preproc_pipe_std = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preproc_pipe_power = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('power', PowerTransformer())
    ])    
    
    if preproc_pipe == 'basic':
        ct = ColumnTransformer(transformers=[
            ('num_standard', preproc_pipe_std, num_not_transform),
            ('num_transform', preproc_pipe_std, num_transform)
        ], remainder = remainder)
    elif preproc_pipe == 'power':
        ct = ColumnTransformer(transformers=[
            ('num_standard', preproc_pipe_std, num_not_transform),
            ('num_transform', preproc_pipe_power, num_transform)
        ], remainder = remainder)
    else:
        ct = None
    return ct
