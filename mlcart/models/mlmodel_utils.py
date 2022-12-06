from sklearn.model_selection import train_test_split
from sklearn.metrics import *

import numpy as np


def bootstrap_confidence_lasso_ridge_(model, n_iterations, data, targetcol, num_cols, cat_cols):
    """
    get confidence score of the model using bootstrapping.
    args:
        targetcol: target column name
        data: whole cleaned dataframe
        num_cols: List of numerical column names
        cat_cols: List of categorical column names
        n_iterations: number of iteration to run bootstrapping
    return:
        Confidence interval: Lower and upper quantile.
    """
    try:
        scores = []
        mae = []
        for i in range(n_iterations):
            X_train, X_test, y_train, y_test = train_test_split(
                data[num_cols+cat_cols], data[targetcol], test_size=0.2, random_state=i)
            X_train = X_train.fillna(X_train.median())
            X_test = X_test.fillna(X_train.median())
            model = model.fit(X_train, y_train)
            scores.append(model.score(X_test, y_test))
            mae.append(mean_absolute_error(model.predict(X_test), y_test))
        lower_mae, upper_mae = np.quantile(
            mae, q=0.025), np.quantile(mae, q=0.975)
        lower_score, upper_score = np.quantile(
            scores, q=0.025), np.quantile(scores, q=0.975)

        return lower_score, upper_score, lower_mae, upper_mae
    except:
        print("\n Not enough resampled data test data may have some new categories.")
        return 0, 0


def bootstrap_confidence_logistic(model, n_iterations, data, targetcol, num_cols, cat_cols):
    """
    get confidence (Accuracy by DEFAULT) score of the model using bootstrapping.
    args:
        targetcol: target column name
        data: whole cleaned dataframe
        num_cols: List of numerical column names
        cat_cols: List of categorical column names
        n_iterations: number of iteration to run bootstrapping
    return:
        Confidence interval: Lower and upper quantile.
    """
    try:
        scores = []
        for i in range(n_iterations):
            X_train, X_test, y_train, y_test = train_test_split(
                data[num_cols+cat_cols], data[targetcol], test_size=0.2, random_state=i)
            X_train = X_train.fillna(X_train.median())
            X_test = X_test.fillna(X_train.median())
            model = model.fit(X_train, y_train)
            scores.append(model.score(X_test, y_test))
        lower, upper = np.quantile(
            scores, q=0.025), np.quantile(scores, q=0.975)
        return lower, upper
    except:
        print("\n Not enough resampled data test data may have some new categories.")
        return 0, 0
