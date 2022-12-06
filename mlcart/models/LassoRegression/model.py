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


class LassoModel:
    def __init__(self, X_train: pd.DataFrame = None, y_train: pd.DataFrame = None):
        self.X_train = X_train
        self.y_train = y_train

    def fit_lasso_model(self):
        """
        Train a Lasso Model with cross validation having many different alpha parameters.
        L(β) = sum (yi − hat_yi)^2 + lambda * sum |beta_j|
        This function is not differentiable in its parameters. Hence there is no closed form of the estimator.

        args:
            X_train: training data dataframe
            y_train: Series of train target actual values
        return:
            ridge_model:
            alpha_: best alpha
            explanation: 2D sorted numpy array with (features, features coefficients) 
        """
        lasso_model = LassoCV(
            cv=10, alphas=[1e-3, 1e-2, 1e-1, 1, 10, 20]).fit(self.X_train, self.y_train)
        train_predictions = lasso_model.predict(self.X_train)
        train_r2 = lasso_model.score(self.X_train, self.y_train)
        print("Best Regularization Constant chosen by Lasso model:",
              lasso_model.alpha_)
        print("Lasso Training Model Score:", train_r2)
        feat_selected = list(self.X_train.columns[lasso_model.coef_.nonzero()])
        explanation = np.append(np.array(feat_selected).reshape(-1, 1),
                                lasso_model.coef_[lasso_model.coef_.nonzero()].reshape(-1, 1).astype('float'), 1)
        explanation[:, 1] = np.abs(
            explanation[:, 1].astype(np.float64))  # .round(2)
        explanation = explanation[explanation[:, 1].argsort()]
        hyperparameters = {'alpha': lasso_model.alpha_}
        train_mae = mean_absolute_error(self.y_train, train_predictions)
        return lasso_model, hyperparameters, train_predictions, train_r2, explanation, train_mae

    def predict_and_evaluate(self, model, X_test: pd.DataFrame = None, y_test: pd.DataFrame = None):
        """
        Do prediction using the model and then calculate the R2
        args:
            model: fitted model object
            X_test: DataFrame of Test data samples
            y_test: Series of test target actual values
        return:
            predictions: array of test predictions
            test_r2: R2 for test data    
        """
        predictions = model.predict(X_test)
        test_r2 = model.score(X_test, y_test)
        test_mae = mean_absolute_error(y_test, predictions)
        return predictions, test_r2, test_mae


class PredictOutOfSample:
    """
    Class to predict out of samples.
    """

    def __init__(self, model, mordred_descpt):
        """
        args: 
            model: Lasso Model used to train.
            mordred_descpt: calculated mordred descriptors to pass into the model.
        """
        self.mordred_descpt = mordred_descpt
        self.model = model

    def prediction(self):
        """
        Call Lasso Model predict to get predictions for the given descriptors.
        return:
            model: Lasso Model
            pred: predictions for given descriptors
        """
        pred = self.model.predict(self.mordred_descpt)
        return pred
