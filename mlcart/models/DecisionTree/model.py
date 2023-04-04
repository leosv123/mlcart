from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import pickle
import re


from sklearn.metrics import *
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


import warnings
warnings.filterwarnings('ignore')


def nestedcv_decisiontree(model, X_train, y_train, metric, metric_low: bool = False):
    """
    Running Nested Cross Validation for Decision Tree to choose:
    1. max_depth:
    2. min_samples_split:
    """
    # Define a parameters grid
    print("Running Decision Tree NestedCV")
    param_grid = {
        'max_depth': [3, 7, 10, 15],
        'min_samples_split': [2, 3, 7, 5],
    }
    NCV = NestedCV(model=model, params_grid=param_grid,
                   outer_kfolds=5, inner_kfolds=5, n_jobs=-1, cv_options={'metric': metric, 'metric_score_indicator_lower': metric_low})
    NCV.fit(X=X_train, y=y_train)

    chosen_parameters = NCV.best_inner_params_list[np.argmax(NCV.outer_scores)]
    print(chosen_parameters)
    return chosen_parameters


class DecisionTreeClassification:
    def __init__(self, X_train: pd.DataFrame = None, y_train: pd.DataFrame = None):
        self.X_train = X_train
        self.y_train = y_train

    def fit_decision_tree_classification(self):
        """
        Train a Logistic Regression model with cross validation having many different alpha parameters.

        This function is not differentiable in its parameters. Hence there is no closed form of the estimator.

        args:
            X_train: training data dataframe
            y_train: Series of train target actual values
        return:
            ridge_model:
            alpha_: best alpha
            explanation: 2D sorted numpy array with (features, features coefficients) 
        """
        temp_model = DecisionTreeClassifier(
            criterion='entropy', random_state=1)
        hyperparameters = nestedcv_decisiontree(
            temp_model, X_train, y_train, roc_auc_score, False)
        decision_tree = DecisionTreeClassifier(random_state=0, max_depth=hyperparameters['max_depth'],
                                               criterion="entropy",
                                               min_samples_split=hyperparameters['min_samples_split'])
        model = decision_tree.fit(self.X_train, self.y_train)
        train_predictions = model.predict(X_train)

        train_accuracy = model.score(self.X_train, self.y_train)
        train_f1_score = f1_score(self.y_train, train_predictions)
        recall = recall_score(self.y_train, train_predictions)
        precision = precision_score(self.y_train, train_predictions)
        score_roc_auc = roc_auc_score(self.y_train, train_predictions)

        print("Train ROC AUC score:", score_roc_auc)
        explanation = np.append(np.array(
            self.X_train.columns).reshape(-1, 1), model.feature_importances_.reshape(-1, 1), axis=1)
        explanation = explanation[explanation[:, 1].argsort()]
        return model, hyperparameters, train_predictions, train_accuracy, train_f1_score, recall, precision, score_roc_auc, explanation
