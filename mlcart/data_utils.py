import pandas as pd
import numpy as np
import os

from mordred import Calculator, descriptors
from rdkit import Chem

import warnings
warnings.filterwarnings("ignore")


def All_Mordred_descriptors(calc, drugs, descriptors_list):
    """
    Calculate all Mordred Descriptors
    return:
        df: dataframe input X
        dropping_indices: indices of rows to be dropped in both X and Y
    """
    mols = []
    dropping_indices = []
    for i in range(len(drugs)):
        try:
            print(drugs[i])
            mol = Chem.MolFromSmiles(drugs[i])
            if mol:
                mols.append(mol)
            else:
                dropping_indices.append(i)
        except:
            dropping_indices.append(i)
    print(mols)
    calc_desc = list(calc.map(mols))
    calc_desc = [np.array(i.fill_missing(np.nan)) for i in calc_desc]
    df = pd.DataFrame(calc_desc, columns=descriptors_list)
    return df, dropping_indices
