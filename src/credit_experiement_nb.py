from sacred import Experiment
from sacred.observers import SqlObserver
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate, train_test_split
import os
from dotenv import load_dotenv
import argparse
from logger import get_logger


from credit_preproc_ingredient import preproc_ingredient, get_column_transformer
from credit_data_ingredient import data_ingredient, load_data
load_dotenv()


db_url = os.getenv('DB_URL')

_logs = get_logger(__name__)
ex  = Experiment("Credit Experiment",
                 ingredients=[data_ingredient, preproc_ingredient])

ex.logger = _logs
ex.observers.append(SqlObserver(db_url))

@ex.config
def cfg():
    preproc_pipe = "power"
    folds = 5
    scoring = ['neg_log_loss', 'roc_auc', 'f1', 'accuracy', 'precision', 'recall']

    

@ex.capture
def get_pipe(preproc_pipe):

    _logs.info(f'Getting Naive Bayes Pipeline')
    ct = get_column_transformer(preproc_pipe)
    pipe = Pipeline(
        steps  = [
            ('preproc', ct),
            ('clf', GaussianNB())
        ]
    )
    return pipe


@ex.capture
def evaluate_model(pipe, X, Y, folds, scoring):
    _logs.info(f'Evaluating model')
    res_dict = cross_validate(pipe, X, Y, cv = folds, scoring = scoring)
    res = pd.DataFrame(res_dict).assign(run_id = 1)
    return res

@ex.automain
def run():
    _logs.info(f'Running experiment')
    X, Y  = load_data()
    pipe = get_pipe()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
    res = evaluate_model(pipe, X_train, Y_train)
    _logs.info(f'res: {res}')
    
    

if __name__=="__main__":
    ex.run_commandline()