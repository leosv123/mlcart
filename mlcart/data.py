import pandas as pd
import numpy as np
import pickle
import re

from mordred import Calculator, descriptors
from rdkit import Chem
from mlcart.data_utils import All_Mordred_descriptors

import warnings
warnings.filterwarnings('ignore')


class Data:
    """
    Load the data categorize the columns to categorical and numerical
    """

    def __init__(self, filepath: str = None, targetcol: str = None):
        self.data = pd.read_csv(filepath)
        cols_nacount = self.data.isna().sum()
        self.data = self.data.drop(
            list(cols_nacount[cols_nacount > 0].index), axis=1)
        self.targetcol = targetcol

    def get_numericals(self):
        self.num_cols = list(self.data.dtypes[
            (self.data.dtypes == 'int64') | (self.data.dtypes == 'float64') |
            (self.data.dtypes == 'int32') | (self.data.dtypes == 'float32') |
            (self.data.dtypes == 'int16') | (self.data.dtypes == 'float16') |
            (self.data.dtypes == 'int8')].keys())
        if self.targetcol in self.num_cols:
            self.num_cols.remove(self.targetcol)
        self.cat_cols = list(set(self.data.columns).difference(self.num_cols))
        if self.targetcol in self.cat_cols:
            self.cat_cols.remove(self.targetcol)
        return self.num_cols, self.cat_cols

    def get_inputs(self):
        num_cols, cat_cols = self.get_numericals()
        return num_cols, self.targetcol, self.data, cat_cols


class OutOfSampleData:
    def __init__(self, drugs, features_proppred) -> None:
        self.calc = Calculator(descriptors, ignore_3D=False)
        self.descriptors_list = [str(i).split('.')[-1]
                                 for i in list(self.calc.descriptors)]
        self.drugs = drugs
        self.feat_proppred = features_proppred
        print("Number of drugs before:", len(drugs))

    def get_mordred_descriptors(self):
        mordred_descpt, dropping_indices = All_Mordred_descriptors(
            self.calc, self.drugs, self.descriptors_list)
        mordred_descpt = mordred_descpt[self.feat_proppred]
        mordred_descpt = mordred_descpt.fillna(0)
        return mordred_descpt, dropping_indices
