import pandas as pd


class Data:
    """
    Load the data categorize the columns to categorical and numerical
    """

    def __init__(self, filepath: str = None, targetcol: str = None):
        self.data = pd.read_csv(filepath)
        columns = []
        for i in self.data.columns:
            i = i.replace('-','_')
            i = i.replace('(','_')
            i = i.replace(')','_')
            columns.append(i)
        self.data.columns = columns
        aaa = self.data.isna().sum()
        self.data = self.data.drop(list(aaa[aaa>0].index),axis=1)
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
