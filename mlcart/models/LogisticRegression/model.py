import pandas as pd
import numpy as np
import pickle
import re


from sklearn.metrics import *
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score


import warnings
warnings.filterwarnings('ignore')


class LogisticRegressionModel:
    def __init__(self, X_train: pd.DataFrame = None, y_train: pd.DataFrame = None):
        self.X_train = X_train
        self.y_train = y_train

    def fit_logistic_regression(self):
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
        lr_clf_model = LogisticRegressionCV(
            cv=10, penalty="l2", scoring='roc_auc').fit(self.X_train, self.y_train)
        train_predictions = lr_clf_model.predict(self.X_train)
        train_pred_prob_1 = lr_clf_model.predict_proba(self.X_train)[:, 1]
        score_roc_auc = lr_clf_model.score(self.X_train, self.y_train)
        print("Logistic Regression Training Model AUC score:", score_roc_auc)

        train_f1_score = f1_score(self.y_train, train_predictions)
        recall = recall_score(self.y_train, train_predictions)
        precision = precision_score(self.y_train, train_predictions)
        train_accuracy = accuracy_score(self.y_train, train_predictions)

        feat_selected = list(
            self.X_train.columns[lr_clf_model.coef_[0].nonzero()])
        explanation = np.append(np.array(feat_selected).reshape(-1, 1),
                                lr_clf_model.coef_[0][lr_clf_model.coef_[0].nonzero()].reshape(-1, 1).astype('float'), 1)
        explanation[:, 1] = np.abs(explanation[:, 1].astype(np.float64))
        explanation = explanation[explanation[:, 1].argsort()]
        return lr_clf_model, train_predictions, train_pred_prob_1, train_accuracy, train_f1_score, recall, precision, score_roc_auc, explanation

    def predict_and_evaluate(self, model, X_test: pd.DataFrame = None, y_test: pd.DataFrame = None):
        """
        Do prediction using the Logistic model and then calculate the Accuracy
        args:
            model: fitted model object
            X_test: DataFrame of Test data samples
            y_test: Series of test target actual values
        return:
            predictions: array of test predictions
            test_r2: R2 for test data    
        """
        predictions = model.predict(X_test)
        predictions_proba_1 = model.predict_proba(X_test)[:, 1]
        score_roc_auc = model.score(X_test, y_test)
        print("Logistic Regression Test AUC score:", score_roc_auc)
        test_f1_score = f1_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        test_accuracy = accuracy_score(y_test, predictions)
        return predictions, predictions_proba_1, test_accuracy, test_f1_score, recall, precision, score_roc_auc


class PredictOutOfSample:
    """
    Class to predict out of samples.
    """

    def __init__(self, model, mordred_descpt):
        """
        args: 
            model: Logistic Model used to train.
            mordred_descpt: calculated mordred descriptors to pass into the model.
        """
        self.mordred_descpt = mordred_descpt
        self.model = model

    def prediction(self):
        """
        Call Logistic Model predict to get predictions for the given descriptors.
        return:
            model: Logistic Model
            pred: predictions for given descriptors
        """
        pred = self.model.predict(self.mordred_descpt)
        pred_prob_1 = self.model.predict_proba(self.mordred_descpt)[:, 1]
        return pred, pred_prob_1
