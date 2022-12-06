from mlcart.models.LinearRegression.data import *
from mlcart.models.LinearRegression.model import *

from omegaconf import OmegaConf
from argparse import ArgumentParser

import json
import sys
sys.setrecursionlimit(5000)


def run_linearregression(filepath: str = None, targetcol: str = None, type: str = "regression"):
    """Load data Split it train test"""
    data_obj = Data(filepath, targetcol)
    num_cols, targetcol, data, cat_cols = data_obj.get_inputs()
    x_train, x_test, y_train, y_test = train_test_split(
        data[num_cols+cat_cols], data[targetcol], test_size=0.2, random_state=0)
    x_train[targetcol] = y_train
    x_test[targetcol] = y_test

    # Before model without any stat test or feature selection
    model = smf.ols(targetcol+'~'+'+'.join(num_cols+cat_cols),
                    data=x_train).fit()
    print("BEFORE:")
    beforemodel = {}
    tansformation = None
    beforemodel['residuals_plot'] = {'x': list(model.fittedvalues.values), 'y': list(model.resid.values), 'title': 'Residuals Plot',
                                     'xlabel': 'Fitted Values', 'ylabel': 'Residuals'}
    bp_test_result = LinearModel.bp_test_fn(model)
    if bp_test_result['LM-Test p-value'] < 0.05:
        if len(data[data[targetcol] <= 0]) > 0:
            info = "Presence of Heteroscedasticity is significant. But cannot apply log transformation on target column as it consists of negative or 0 values."
        else:
            info = "Presence of Heteroscedasticity is significant. Hence, applying log transformation on target column."
            data[targetcol] = np.log(data[targetcol])
            tansformation = "log"
    else:
        info = "Presence of Heteroscedasticity is not significant."
    beforemodel['breusch_pegan_test'] = {
        'pvalue': bp_test_result['LM-Test p-value'], 'info': info}
    predictions = model.predict(x_test[num_cols+cat_cols])
    score_r2 = LinearModel.adjusted_r2(predictions, x_test, targetcol)
    beforemodel['test_adj_r2'] = score_r2

    # start statistical tests
    # num_cols, cat_cols, vif_removal = LinearModel.removal_on_vif(
    #     targetcol, data, num_cols, cat_cols)
    #beforemodel['vif_removal'] = vif_removal
    # try:
    # outliers = LinearModel.influential_points(
    #     targetcol, num_cols, cat_cols, data)
    # beforemodel['outliers_index'] = outliers
    # data = data.drop(outliers).reset_index()
    # print("length of data after removing outliers:", len(data))
    # except:
    #     outliers = ["not able to calculate"]
    num_cols, cat_cols, pvalue_removal = LinearModel.removal_on_pvalue(
        targetcol, num_cols, cat_cols, data)
    beforemodel['pvalue_removal'] = pvalue_removal
    num_cols, cat_cols, anovatyp1_removal = LinearModel.anovatyp1_removal(
        targetcol, num_cols, cat_cols, data)
    beforemodel['anovatyp1_removal'] = anovatyp1_removal

    # Finalmodel
    final_linearmodel = {}
    model, explanation, num_cols, cat_cols = LinearModel.finalmodel(
        targetcol, num_cols, cat_cols, x_train)
    mse = model.mse_resid
    msr = model.mse_model
    predictions = model.predict(x_test[num_cols+cat_cols])
    score_r2 = LinearModel.adjusted_r2(predictions, x_test, targetcol)
    if score_r2 < 0 or score_r2 > 1:
        score_r2 = r2_score(predictions, x_test[targetcol])
    n_iterations = 200
    lower, upper = LinearModel.bootstrap_confidence(
        n_iterations, targetcol, num_cols, cat_cols, data)
    conf_interval, pred_interval = LinearModel.ci_pi(
        targetcol, num_cols, cat_cols, data, mse, model)
    print("AFTER:")
    print(filepath.split('/')[-1])
    final_linearmodel = {"datainfo": {'filepath': filepath.split('/')[-1],
                                      'numoffeats': len(num_cols+cat_cols), 'cat_cols': cat_cols, 'numerical_cols': num_cols, 'numofobservations': len(data)}}
    bp_test_result = LinearModel.bp_test_fn(model)
    final_linearmodel['explanation'] = {'x': list(explanation.values), 'y': list(explanation.index), 'title': 'Explanation based on absolute tvalue',
                                        'xlabel': 'Contribution magnitude', 'ylabel': 'Features'}
    final_linearmodel['test_adj_r2'] = score_r2
    final_linearmodel['train_adj_r2'] = model.rsquared_adj
    final_linearmodel['train_mse'] = mse
    final_linearmodel['train_msr'] = msr
    if bp_test_result['LM-Test p-value'] < 0.05:
        info = "Presence of Heteroscedasticity is significant still. Try other ML algorithms. Sometimes threshold can be little less its fine. check properly."
    else:
        info = "Presence of Heteroscedasticity is not significant"
    final_linearmodel['bootstrap_confidence'] = {
        'lower_bound': lower, 'upper_bound': upper}
    final_linearmodel['breusch_pegan_test'] = {
        'pvalue': bp_test_result['LM-Test p-value'], 'info': info}
    final_linearmodel['ci'] = conf_interval
    final_linearmodel['residuals_plot'] = {'x': list(model.fittedvalues.values), 'y': list(model.resid.values), 'title': 'Residuals Plot',
                                           'xlabel': 'Fitted Values', 'ylabel': 'Residuals'}

    data['fit_y'] = model.fittedvalues
    data_new = data.sort_values(by=[targetcol]).reset_index()
    if tansformation == "log":
        data_new[targetcol] = np.exp(data_new[targetcol])
    print("pred", data_new['fit_y'].max())
    print("actucal", data_new[targetcol].max())
    final_linearmodel['modelperformance'] = {
        'x': list(data_new.index),
        'y_pred': list(data_new['fit_y']),
        'y_true': list(data_new[targetcol]),
        'title': "Model Performance Graph",
        'xlabel': "Sorted by observation Index(target column)",
        'ylabel': "Predictions"
    }
    return beforemodel, final_linearmodel


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

    before, finalconfig_dict = run_linearregression(filepath, targetcol)

    json_object = json.dumps(finalconfig_dict, indent=4)
    with open("lr_output.json", "w") as outfile:
        outfile.write(json_object)
