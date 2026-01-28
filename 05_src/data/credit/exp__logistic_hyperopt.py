import mlflow
from mlflow.models import infer_signature

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import get_scorer

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe


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
  """
  Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

  This function checks if an experiment with the given name exists within MLflow.
  If it does, the function returns its ID. If not, it creates a new experiment
  with the provided name and returns its ID.

  Parameters:
  - experiment_name (str): Name of the MLflow experiment.

  Returns:
  - str: ID of the existing or newly created MLflow experiment.

  Ref: https://mlflow.org/docs/latest/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/hyperparameter-tuning-with-child-runs#how-will-we-use-the-experiment_id
  """

  if experiment := mlflow.get_experiment_by_name(experiment_name):
      return experiment.experiment_id
  else:
      return mlflow.create_experiment(experiment_name)


def run_cv(params, 
           folds = 5, 
           model_name='credit_logistic_hyperopt',
           test_size = 0.2,
           scoring = ['neg_log_loss'],
           random_state = None, 
           tags = None):
    
    pipe = get_pipe()
    X, Y = load_data(CREDIT_FILE)

    if isinstance(scoring, str):
        _logs.info(f'Converting scoring to list')
        scoring = [scoring]
    if tags is None:
        _logs.info(f'No tags provided, using empty dict')
        tags = {}
    
    with mlflow.start_run(nested=True):
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
        _logs.info('Logging cross-validation metrics to MLflow')
        mlflow.log_metrics(mean_res_cv)
        
        # pipe.fit(X_train, Y_train)
        # signature = infer_signature(X_train, pipe.predict(X_train))
        # _logs.info('Logging model to MLflow')
        # model_info = mlflow.sklearn.log_model(
        #     sk_model=pipe, 
        #     signature = signature,
        #     pip_requirements="requirements.txt",
        #     input_example = X_train.head(5),
        #     artifact_path="model"
        # )
    
        return {'loss': -mean_res_cv['test_neg_log_loss'], 'status': STATUS_OK, 'model': pipe, 'params': params}





def hyperparam_opt(scoring = ['neg_log_loss', 'balanced_accuracy', 'f1'], 
                   folds = 5, 
                   random_state = 42, 
                   experiment_name='credit_hyperopt_logistic'):
    _logs.info(f'Running experiment')

    space = {
        'preproc__num_standard__imputer__add_indicator': hp.choice("preproc__num_standard__imputer__add_indicator", [False, True]),
        'clf__C': hp.uniform("clf__C", 0.1,  1.0),
        'clf__class_weight': hp.choice("clf__class_weight", [None, 'balanced']),
        'clf__penalty': hp.choice("clf__penalty", ['l2']),
        'clf__random_state': hp.choice("clf__random_state", [random_state]),
        'clf__solver': hp.choice("clf__solver", ['lbfgs', 'liblinear'])
    }

    _logs.info(f'Creating experiment {experiment_name}')
    experiment_id = get_or_create_experiment(experiment_name)
    _logs.info(f'Setting experiment ID to {experiment_id}')
    mlflow.set_experiment(experiment_id=experiment_id)

    with mlflow.start_run():
        trials = Trials()
        best = fmin(fn=lambda params: run_cv(params, tags={'optimizer': 'hyperopt'}, scoring = ['neg_log_loss', 'balanced_accuracy', 'f1']),
            space=space,
            algo=tpe.suggest,
            max_evals=10,
            trials=trials
        )

        best_run = sorted(trials.results, key= lambda x: x['loss'])[0]

        mlflow.log_params(best)
        mlflow.log_metric("test_neg_log_loss", -best_run['loss'])
        # mlflow.sklearn.log_model(best_run['model'], 
        #                          artifact_path="model", 
        #                          registered_model_name="CreditLogisticModel")


if __name__=="__main__":
    hyperparam_opt()