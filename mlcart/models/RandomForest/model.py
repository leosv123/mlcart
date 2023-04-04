import pandas as pd
import numpy as np
import pickle
import re


from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')

from run_nestedcv import NestedCV

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston, load_iris, load_breast_cancer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold

# When using Random Search, we get a user warning with this little number of hyperparameters
# Suppress it
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


def nestedcv_rf(model, X_train, y_train, metric, metric_low:bool=False):
    """
    Running Nested Cross Validation for RF to choose:
    1. max_depth:
    2. min_samples_split:
    3. n_estimators:
    """
    #Define a parameters grid
    print("Running RF NestedCV")
    param_grid = {
         'n_estimators':[100,200],
         'max_depth': [5,7],
         'min_samples_split':[5,7], 
    }
    NCV = NestedCV(model=model, params_grid=param_grid,
                   outer_kfolds=5, inner_kfolds=5, n_jobs = -1, cv_options={'metric':metric,'metric_score_indicator_lower':metric_low})
    NCV.fit(X=X_train,y=y_train)

    chosen_parameters = NCV.best_inner_params_list[np.argmax(NCV.outer_scores)]
    print(chosen_parameters)
    return chosen_parameters



class RFClassification:
    def __init__(self, X_train:pd.DataFrame=None, y_train:pd.DataFrame=None):
        self.X_train = X_train
        self.y_train = y_train
        
    def fit_rf_classification(self):
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
        temp_model = RandomForestClassifier(criterion='entropy', random_state=1)
        hyperparameters = nestedcv_rf(temp_model, X_train, y_train, roc_auc_score, False)
        rf = RandomForestClassifier(n_estimators=hyperparameters['n_estimators'],random_state=0, max_depth=hyperparameters['max_depth'], 
                                               criterion="entropy",
                                               min_samples_split=hyperparameters['min_samples_split'])
        model = rf.fit(self.X_train, self.y_train)
        train_predictions = model.predict(X_train)

        train_accuracy = model.score(self.X_train,self.y_train)
        train_f1_score = f1_score(self.y_train, train_predictions)
        recall = recall_score(self.y_train, train_predictions)
        precision = precision_score(self.y_train, train_predictions)
        score_roc_auc = roc_auc_score(self.y_train, train_predictions)
        
        print("Train ROC AUC score:", score_roc_auc)
        explanation = np.append(np.array(self.X_train.columns).reshape(-1,1), model.feature_importances_.reshape(-1,1),axis=1)
        explanation = explanation[explanation[:,1].argsort()]
        return model, hyperparameters, train_predictions, train_accuracy, train_f1_score, recall, precision, score_roc_auc, explanation