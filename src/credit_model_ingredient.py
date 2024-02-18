from sacred import Ingredient
from dotenv import load_dotenv

from logger import get_logger
load_dotenv()
_logs = get_logger(__name__)

model_ingredient = Ingredient('model_ingredient')
model_ingredient.logger = _logs

@model_ingredient.config
def cfg():
    model = 'NaiveBayes'


@model_ingredient.capture
def get_model(model):
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
    if model == 'MLP':
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier()
    else:
        return None

