import mlflow
from mlflow.models import infer_signature

import pandas as pd


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import get_scorer
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
    )
    X = df.drop(columns = ['delinquency'])
    Y = df['delinquency']
    return X, Y
    

def get_pipe():
    _logs.info(f'Getting Model Pipeline')
    
    num_std_cols = ['num_30_59_days_late', 
                     'num_60_89_days_late', 
                     'num_90_days_late',
                     'num_open_credit_loans', 
                     'num_real_estate_loans',
                     'age', 'num_dependents']
    num_pow_cols = ['revolving_unsecured_line_utilization', 
                    'monthly_income', 
                    'debt_ratio']
    
    preproc_pipe_std = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    
    preproc_pipe_power = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('power', PowerTransformer())
    ])    

    ct = ColumnTransformer(transformers=[
            ('num_standard', preproc_pipe_std, num_std_cols),
            ('num_pow_cols', preproc_pipe_power, num_pow_cols)
        ], remainder = "passthrough")

    pipe = Pipeline(
        steps  = [
            ('preproc', ct),
            ('clf', LogisticRegression())
        ]
    )
    return pipe


def evaluate_model(pipe, X, Y,
                   scoring, 
                   params,
                   folds = 5,
                   test_size = 0.2,  
                   random_state = None):
    _logs.info(f'Evaluating model using {scoring} and {folds} folds')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                        test_size = test_size, 
                                                        random_state = random_state)
    pipe.set_params(**params)
    res_cv = cross_validate(pipe, X_train, Y_train, cv = folds, scoring = scoring)
    _logs.debug(f'CV results: {res_cv}')
    register_experiment(pipe, params, X_train, Y_train, res_cv)


def register_experiment(pipe, params, X_train, Y_train, res_cv):
    _logs.info('Logging model')
    mlflow.set_tracking_uri(uri=MLFLOW_URI)
    mlflow.set_experiment("MLFlow first experiment")
    with mlflow.start_run():
        mlflow.log_params(params)

        mlflow.log_metric("fit_time", res_cv['fit_time'].mean())
        mlflow.log_metric("score_time", res_cv['score_time'].mean())
        mlflow.log_metric("mean_score", res_cv['test_score'].mean())
        mlflow.set_tag("CV performance", "Basic LR model for credit data")

        pipe.fit(X_train, Y_train)
        signature = infer_signature(X_train, pipe.predict(X_train))

        model_info = mlflow.sklearn.log_model(
            sk_model=pipe, 
            artifact_path='credit_simple', 
            signature = signature,
            input_example = X_train,
            registered_model_name='tracking-credit-initial'
        )

def run(scoring = 'neg_log_loss', folds = 5, random_state = 42):
    _logs.info(f'Running experiment')
    X, Y  = load_data()
    pipe = get_pipe()
    params = {
        
        'preproc__num_standard__imputer__add_indicator': False,
        'preproc__num_standard__imputer__copy': True,
        'preproc__num_standard__imputer__strategy': 'median',
        'preproc__num_standard__scaler__with_mean': True,
        'preproc__num_standard__scaler__with_std': True,
        'preproc__num_pow_cols__imputer__strategy': 'median',
        'preproc__num_pow_cols__scaler__with_mean': True,
        'preproc__num_pow_cols__scaler__with_std': True,
        'preproc__num_pow_cols__power__method': 'yeo-johnson',
        'preproc__num_pow_cols__power__standardize': True,
        'clf__C': 1.0,
        'clf__class_weight': None,
        'clf__fit_intercept': True,
        'clf__intercept_scaling': 1,
        'clf__l1_ratio': None,
        'clf__max_iter': 100,
        'clf__penalty': 'l2',
        'clf__random_state': random_state,
        'clf__solver': 'lbfgs'
    }
    res_cv = evaluate_model(pipe, X, Y, 
                            scoring = "neg_log_loss", 
                            params = params,
                            folds = folds, 
                            random_state = random_state)

   
if __name__=="__main__":
    run()