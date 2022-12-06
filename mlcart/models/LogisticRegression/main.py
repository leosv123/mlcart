import json
from argparse import ArgumentParser
from omegaconf import OmegaConf
import pandas as pd
import numpy as np
import pickle
import re


from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score

from mlcart.models.LogisticRegression.data import *
from mlcart.models.LogisticRegression.model import *
from mlcart.models.mlmodel_utils import *

import warnings
warnings.filterwarnings('ignore')


def run_logistic_regression(filepath: str = None, targetcol: str = 'property', type="classification"):
    data_obj = Data(filepath, targetcol)
    num_cols, targetcol, data, cat_cols = data_obj.get_inputs()
    data = data.fillna(data.median())
    X_train, X_test, y_train, y_test = train_test_split(
        data[num_cols+cat_cols], data[targetcol], test_size=0.2, random_state=0)

    model = LogisticRegressionModel(X_train, y_train)
    logistic_model, train_predictions, train_prob_1, train_accuracy, train_f1_score, train_recall, train_precision, train_score_roc_auc, explanation = model.fit_logistic_regression()
    test_predictions, test_prob_1, test_accuracy, test_f1_score, test_recall, test_precision, test_score_roc_auc = model.predict_and_evaluate(
        logistic_model, X_test, y_test)
    print("Bootstrapping running")
    lower_auc, upper_auc = bootstrap_confidence_logistic(
        logistic_model, 2, data, targetcol, num_cols, cat_cols)
    print(f"CI:{lower_auc,upper_auc}\n")

    model_file = 'logistic_model.pkl'
    pickle.dump(logistic_model, open(model_file, 'wb'))

    train_data = X_train.copy()
    train_data['property'] = y_train
    train_data['predicted'] = train_predictions
    train_data['predicted_prob'] = train_prob_1
    train_file = "train_file.csv"
    train_data.to_csv(train_file, index=None)

    test_data = X_test.copy()
    test_data['property'] = y_test
    test_data['predicted'] = test_predictions
    test_data['predicted_prob'] = test_prob_1
    test_file = "test_data.csv"
    test_data.to_csv(test_file, index=None)

    finalmodel = {"mlalgorithm": "Logistic Regression", "hyperparameters": {'key': 0},
                  "model_file": model_file,
                  "input_features": list(data.columns)[:-1],
                  "train_prediction_data": train_file,
                  "train_metric": [{'metric': 'Accuracy', 'score': train_accuracy},
                                   {'metric': 'F1 Score', 'score': train_f1_score},
                                   {'metric': 'Recall', 'score': train_recall},
                                   {'metric': 'Precision',
                                       'score': train_precision},
                                   {'metric': 'ROC AUC score', 'score': train_score_roc_auc}],
                  "explanation": {'x': list(explanation[:, 0][-20:]), 'y': list(explanation[:, 1][20:]),
                                  'title': 'Which Features are responsible?', 'xlabel': 'Contribution magnitude',
                                  'ylabel': 'Features'},
                  "test_prediction_data": test_file,
                  "test_metric": [{'metric': 'Accuracy', 'score': test_accuracy},
                                  {'metric': 'F1 Score', 'score': test_f1_score},
                                  {'metric': 'Recall', 'score': test_recall},
                                  {'metric': 'Precision', 'score': test_precision},
                                  {'metric': 'ROC AUC score', 'score': test_score_roc_auc}],
                  "ci": [{'metric': 'ROC AUC score', 'interval': [lower_auc, upper_auc]}]
                  }
    return finalmodel


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        required=True, help="provide the config file")
    args = parser.parse_args()
    # get configuration
    configFile = OmegaConf.load(args.config)
    modelname = configFile.config.modelname
    filepath = configFile.config.filepath
    targetcol = configFile.config.targetcol

    finalconfig_dict = run_logistic_regression(filepath, targetcol)

    print(finalconfig_dict)
    json_object = json.dumps(finalconfig_dict, indent=4)
    with open("logistic_output.json", "w") as outfile:
        print(json_object)
        outfile.write(json_object)
