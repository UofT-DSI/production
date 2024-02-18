from sacred import Experiment
from sacred.observers import SqlObserver

import pandas as pd


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
import os
from dotenv import load_dotenv

from logger import get_logger


from credit_preproc_ingredient import preproc_ingredient, get_column_transformer
from credit_data_ingredient import data_ingredient, load_data
from credit_db_ingredient import db_ingredient, df_to_sql
from credit_model_ingredient import model_ingredient, get_model, get_param_grid
load_dotenv()


db_url = os.getenv('DB_URL')

_logs = get_logger(__name__)
ex  = Experiment("Credit Experiment",
                 ingredients=[data_ingredient, preproc_ingredient, db_ingredient, model_ingredient])

ex.logger = _logs
ex.observers.append(SqlObserver(db_url))

@ex.config
def cfg():
    preprocessing = "power"
    model = 'NaiveBayes'
    folds = 5
    scoring = ['neg_log_loss', 'roc_auc', 'f1', 'accuracy', 'precision', 'recall']
    refit='neg_log_loss'


@ex.capture
def get_pipe(preprocessing, model):

    _logs.info(f'Getting {preprocessing} prepocessing and {model} classifier pipeline.')
    ct = get_column_transformer(preprocessing)
    clf = get_model(model)
    pipe = Pipeline(
        steps  = [
            ('preproc', ct),
            ('clf', clf)
        ]
    )
    return pipe


@ex.capture
def grid_search(pipe, param_grid, X, Y, folds, scoring, refit):
    _logs.info(f'Tuning model')
    gs = GridSearchCV(pipe, param_grid, scoring=scoring, cv = folds, refit = refit)
    gs.fit(X, Y)
    res_dict = gs.cv_results_
    res = pd.DataFrame(res_dict)
    return res

@ex.capture
def res_to_sql(res, model, preprocessing, _run):
    _logs.info(f'Writing results to db')
    res_out = (res.drop(columns=['params'])
                    .assign(model = model, 
                         preprocessing = preprocessing, 
                         run_id = _run._id))
    df_to_sql(df = res_out, 
              table_name = f"model_tuning_cv_{model}",
              if_exists = 'append')

@ex.automain
def run(preprocessing, model):
    _logs.info(f'Running experiment')
    X, Y  = load_data()
    pipe = get_pipe(preprocessing, model)
    param_grid = get_param_grid(model)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
    res = grid_search(pipe, param_grid, X_train, Y_train)   
    res_to_sql(res)
    _logs.info(res)
   
if __name__=="__main__":
    ex.run_commandline()
