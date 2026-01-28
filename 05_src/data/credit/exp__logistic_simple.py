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


from credit.data import load_data
from utils.logger import get_logger

mlflow.set_tracking_uri("http://localhost:5001")

load_dotenv(override=True)
CREDIT_FILE = os.getenv("CREDIT_DATA")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")

_logs = get_logger(__name__)

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

def get_or_create_experiment(experiment_name):
    _logs.info(f'Getting or creating experiment: {experiment_name}')
    _logs.info(f'Checking if experiment {experiment_name} exists')
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        _logs.info(f'Experiment {experiment_name} not found, creating it')
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        _logs.info(f'Experiment {experiment_name} found')
        experiment_id = experiment.experiment_id
    return experiment_id


def run_experiment_cv(pipe, 
                      params, 
                      X, Y, 
                      folds = 5, 
                      experiment_name=None,
                      model_name="credit_logistic_simple",
                      test_size = 0.2,
                      scoring = ['neg_log_loss'],
                      mlflow_uri = MLFLOW_URI,
                      random_state = None, 
                      tags = {'optimizer': 'NA'}):
    
    _logs.info(f'Starting experiment: {experiment_name}')

    if isinstance(scoring, str):
        _logs.info(f'Converting scoring to list')
        scoring = [scoring]
    if tags is None:
        _logs.info(f'No tags provided, using empty dict')
        tags = {}

    _logs.info(f'Creating experiment {experiment_name}')
    experiment_id = get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)
    
    with mlflow.start_run():
        _logs.info('Logging parameters and tags')
        mlflow.log_params(params)
        mlflow.log_params({
            'folds': folds,
            'test_size': test_size,
            'random_state': random_state,
            'scoring': scoring
        })
        mlflow.set_tags(tags)
        _logs.info('Starting cross-validation')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                        test_size = test_size, 
                                                        random_state = random_state)
        pipe.set_params(**params)
        res_cv = cross_validate(pipe, X_train, Y_train, cv = folds, scoring = scoring, return_train_score = True)
        mean_res_cv = pd.DataFrame(res_cv).mean().to_dict()
        mlflow.log_metrics(mean_res_cv)
        
        pipe.fit(X_train, Y_train)
        # signature = infer_signature(X_train, pipe.predict(X_train))

        # model_info = mlflow.sklearn.log_model(
        #     sk_model=pipe, 
        #     signature = signature,
        #     artifact_path = "model", 
        #     input_example = X_train.head(5),
        #     registered_model_name=model_name
        # )

def single_run(scoring = 'neg_log_loss', folds = 5, random_state = 42):
    _logs.info(f'Running experiment')
    pipe = get_pipe()
    X, Y = load_data(CREDIT_FILE)
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

    run_experiment_cv(pipe, 
                      params, 
                      X, Y, 
                      folds = 5, 
                      experiment_name='credit_single_run_logistic',
                      model_name=None,
                      test_size = 0.2,
                      scoring = ['neg_log_loss', 'accuracy', 'f1'],
                      mlflow_uri = MLFLOW_URI,
                      random_state = None, 
                      tags = None)




   
if __name__=="__main__":
    single_run()