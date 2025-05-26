import numpy as np
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from utils.logger import get_logger

_logs = get_logger(__name__)

mlflow.set_tracking_uri("http://localhost:5001")

if __name__ == "__main__":
    _logs.info('Starting ML flow test.')

    mlflow.set_experiment("mlflow_test_experiment")
    with mlflow.start_run():
        try:
            X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
            y = np.array([0, 0, 1, 1, 1, 0])
            _logs.info(f'Created X: {X} and y: {y}')
            lr = LogisticRegression()
            lr.fit(X, y)
            _logs.info(f'Fitted model: {lr}')
            score = lr.score(X, y)
            _logs.info(f"Score: {score}")
            mlflow.log_metric("score", score)
            predictions = lr.predict(X)
            _logs.info(f'Predictions: {predictions}')
            signature = infer_signature(X, predictions)
            _logs.info('Logging model.')
            mlflow.sklearn.log_model(lr, "model", signature=signature)
            _logs.info(f"Model saved in run {mlflow.active_run().info.run_uuid}")
        except Exception as e:
            _logs.error(f"Error during ML flow test: {e}")