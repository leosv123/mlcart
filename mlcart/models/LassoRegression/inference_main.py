
from mlcart.models.LassoRegression.data import *
from mlcart.models.LassoRegression.model import *

import json

def predict_samples(drugs, model_file, input_features):
    model = pickle.load(open(model_file, 'rb'))
    data = OutOfSampleData(drugs, input_features)
    mordred_descpt, dropping_indices = data.get_mordred_descriptors()

    pred_obj = PredictOutOfSample(model, mordred_descpt)
    predictions = pred_obj.prediction()
    print("Predictions for given molecules:", predictions)
    invalid_drugs_list = [drugs[i] for i in dropping_indices]
    valid_drugs = set(drugs)-set(invalid_drugs_list)
    return {'valid_drugs':sorted(list(valid_drugs)), 'predictions': predictions.tolist(), 'invalid_drugs':invalid_drugs_list}


if __name__ == "__main__":
    drugs = ["CCCCNBr", "Oc1cc(O)c2c(c1)OC(c1ccc(O)c(O)c1)C(O)C2",'xxx','aaaa', "C/C=C\C#CCC/C=C\C=C\C(=O)NCC(C)C",
    "COc1ccc2c3c1O[C@H]1[C@@H](O)C=C[C@H]4[C@@H](C2)N(C)CC[C@]314"]
    drugs = sorted(drugs)
    with open("lasso_output.json", 'r') as f:
        data = json.load(f)

    input_features = data["input_features"]
    model_file = data['model_file']
    print(predict_samples(drugs, model_file, input_features))
    