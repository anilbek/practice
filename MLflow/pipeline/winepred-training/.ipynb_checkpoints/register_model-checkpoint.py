import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

import mlflow.sklearn


def acquire_data():
    wine = pd.read_csv('winequality-red.csv')
    return wine


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    with mlflow.start_run():

        data = acquire_data()

        WINDOW_SIZE = 14
        
        X = data.drop('quality', axis=1)
        y = data.quality
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        gbm = GradientBoostingRegressor(n_estimators=500, subsample=0.9, max_depth=12, learning_rate=0.1, random_state=42)

        svr = SVR(C=10.0, gamma=0.3, epsilon=0.0001)

        blender = VotingRegressor(estimators=[('gbm', gbm), ('svm', svr)],n_jobs=-2)

        blender.fit(X_train, y_train)

        preds = blender.predict(X_test)
        
        mlflow.sklearn.log_model(blender, "model_blend")
        
        mae = mean_absolute_error(y_test, np.round(preds))
        
        print(f"test mean absolute error for blender: {mae:0.3}")
        
        mlflow.log_metric("mae", mean_absolute_error(y_test, np.round(preds))