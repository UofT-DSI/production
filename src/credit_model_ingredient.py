from sacred import Ingredient
from dotenv import load_dotenv

from logger import get_logger
import json
import os
load_dotenv()
_logs = get_logger(__name__)

model_ingredient = Ingredient('model_ingredient')
model_ingredient.logger = _logs

@model_ingredient.config
def cfg():
    pass


@model_ingredient.capture
def get_model(model):
    '''
    Given a name, return a model.
    '''
    _logs.info(f'Getting model {model}')
    if model == 'NaiveBayes':
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()
    if model == 'LogisticRegression':
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression()
    if model == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier()
    if model == 'SVM':
        from sklearn.svm import SVC
        return SVC()
    if model == 'NeuralNet':
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier()
    else:
        return None

def get_param_grid(model):
    '''
    Given a name, return a parameter grid. Param grids are stored in json files.
    '''
    _logs.info(f'Getting parameter grid for {model}')
    file = None
    if model == 'LogisticRegression':
        file = os.getenv("LOGISTIC_REGRESSION_PG")
    if model == 'RandomForest':
        file = os.getenv("RANDOM_FOREST_PG")
    if model == 'SVM':
        file = os.getenv("SVM_PG")
    if model == 'NeuralNet':
        file = os.getenv("NEURAL_NET_PG")

    if file is not None:
        with open(file) as f:
            return json.load(f)
    else:
        return None
        
