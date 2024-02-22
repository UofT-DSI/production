from sacred import Experiment
from sacred.observers import SqlObserver

import pandas as pd
import pickle

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
    '''
    Main experiment config.
    '''
    preprocessing = "power"
    model = 'LogisticRegression'
    folds = 5
    scoring = ['neg_log_loss', 'roc_auc', 'f1', 'accuracy', 'precision', 'recall']
    refit='neg_log_loss'


@ex.capture
def get_pipe(preprocessing, model):
    '''
    Main pipeline builder: gets a preprocessing name and a classifier name, returns a pipeline.
    '''
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
    '''Perform grid search on a pipeline given a parameter grid and data.'''
    _logs.info(f'Tuning model')
    gs = GridSearchCV(pipe, param_grid, scoring=scoring, cv = folds, refit = refit)
    gs.fit(X, Y)
    _logs.info(f'Best score: {gs.best_score_}')
    _logs.info(f'Best params: {gs.best_params_}')
    res_dict = gs.cv_results_
    res = pd.DataFrame(res_dict)
    pipe_best = gs.best_estimator_
    return res, pipe_best

@ex.capture
def res_to_sql(res, model, preprocessing, _run):
    '''
    Push CV results to DB.
    '''
    _logs.info(f'Writing results to db')
    res_out = (res.drop(columns=['params'])
                    .assign(model = model, 
                         preprocessing = preprocessing, 
                         run_id = _run._id))
    df_to_sql(df = res_out, 
              table_name = f"model_tuning_cv_{model}",
              if_exists = 'append')

@ex.capture
def pickle_model_artifact(pipe, model, preprocessing, _run):
    '''
    Save model object to disk and add it as an artifact to the experiment run.
    '''

    _logs.info(f'Pickling model artifact')
    
    artifacts_dir = os.getenv('ARTIFACTS_DIR')
    os.makedirs(artifacts_dir, exist_ok=True)
    
    outpath = os.path.join(
        artifacts_dir, 
        f"model_{model}_{preprocessing}_{_run._id}.pkl")
    
    with open(outpath, 'wb') as f:
        pickle.dump(pipe, f)
    # Add artifact to experiment run
    _run.add_artifact(outpath)
    
    _logs.info(f'Pickled model artifact to {outpath}')
    


@ex.main
def run(preprocessing, model):
    '''Main function: runs the experiment.'''
    _logs.info(f'Running experiment')
    X, Y  = load_data()
    pipe = get_pipe(preprocessing, model)
    param_grid = get_param_grid(model)
    if param_grid is not None:
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, 
            test_size = 0.2)
        res, pipe_best = grid_search(pipe, param_grid, X_train, Y_train)   
        _logs.info(f'Optimization results {res.shape}')
        res_to_sql(res)
        pickle_model_artifact(pipe_best, model, preprocessing)
        
    else:
        _logs.warning(f'Parameter grid is None for {model}')
   
if __name__=="__main__":
    ex.run_commandline()
