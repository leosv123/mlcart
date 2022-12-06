from sklearn.model_selection import train_test_split

from omegaconf import OmegaConf
from argparse import ArgumentParser

from mlcart.models.LassoRegression.data import *
from mlcart.models.LassoRegression.model import *
from mlcart.models.mlmodel_utils import *

import json


def run_lassoregression(filepath: str = None, targetcol: str = None, type: str = "regression"):
    """Load data Split it train test"""

    data_obj = Data(
        filepath, targetcol)
    num_cols, targetcol, data, cat_cols = data_obj.get_inputs()
    # data = data.fillna(data.median())
    X_train, X_test, y_train, y_test = train_test_split(
        data[num_cols+cat_cols], data[targetcol], test_size=0.2, random_state=0)

    model = LassoModel(X_train, y_train)
    lasso_model, hyperparameters, lasso_train_predictions, lasso_train_r2, lasso_explanation, train_mae = model.fit_lasso_model()
    lasso_test_predictions, lasso_test_r2, test_mae = model.predict_and_evaluate(
        lasso_model, X_test, y_test)
    print(f"lasso_test_r2:{lasso_test_r2}")

    lower_score, upper_score, lower_mae, upper_mae = bootstrap_confidence_lasso_ridge_(
        lasso_model, 10, data, targetcol, num_cols, cat_cols)
    print(f"CI R2:{lower_score,upper_score}")
    print(f"CI MAE:{lower_mae,upper_mae}")

    model_file = 'lasso_model.pkl'
    pickle.dump(lasso_model, open(model_file, 'wb'))

    train_data = X_train.copy()
    train_data['property'] = y_train
    train_data['predicted'] = lasso_train_predictions
    train_file = "train_file.csv"
    train_data.to_csv(train_file, index=None)

    test_data = X_test.copy()
    test_data['property'] = y_test
    test_data['predicted'] = lasso_test_predictions
    test_file = "test_data.csv"
    test_data.to_csv(test_file, index=None)

    model_performance = np.append(np.array(
        lasso_test_predictions).reshape(-1, 1), np.array(y_test).reshape(-1, 1), 1)
    model_performance = model_performance[model_performance[:, 1].argsort()]

    finalmodel = {"mlalgorithm": "Lasso Regression",
                  "model_file": model_file,
                  "input_features": data.columns.tolist()[:-1],
                  "hyperparameters": hyperparameters,
                  "train_prediction_data": train_file,
                  "train_metric": [{"metric": "R2", "score": lasso_train_r2},
                                   {"metric": "MAE", "score": train_mae}],
                  "explanation": {'x': lasso_explanation[:, 0][-20:].tolist(), 'y': lasso_explanation[:, 1][-20:].tolist(),
                                  'title': 'Which Features are responsible?',
                                  'xlabel': 'Contribution magnitude', 'ylabel': 'Features'},
                  "test_metric": [{"metric": "R2", "score": lasso_test_r2},
                                  {"metric": "MAE", "score": test_mae}],
                  "test_predictions": test_file,
                  "model_performance": {'x': np.arange(len(model_performance)).tolist(),
                                        'y_pred': model_performance[:, 0].tolist(),
                                        'y_true': model_performance[:, 1].tolist(),
                                        'title': 'Model performance based on R2',
                                        'xlabel': 'Observation Index', 'ylabel': 'R2 score'},
                  "ci": [{"metric": "R2 Score", "interval": [lower_score, upper_score]},
                         {"metric": "Mean Absolute Error", "interval": (lower_mae, upper_mae)}]
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

    finalconfig_dict = run_lassoregression(filepath, targetcol)

    json_object = json.dumps(finalconfig_dict, indent=4)
    with open("lasso_output.json", "w") as outfile:
        outfile.write(json_object)
