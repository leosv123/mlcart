import numpy as np
import pandas as pd
import re

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
import scipy.stats as st
import scipy
from statsmodels.stats.diagnostic import het_breuschpagan


from sklearn.utils import resample
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')


class LinearModel:
    def vif(targetcol: str = None, data: pd.DataFrame = None, num_cols: list = None, cat_cols: list = None):
        """
        Calculate the VIF and get the columns having multicollinear features (high VIF).
        args:
            targetcol: target column name
            data: whole cleaned dataframe
            num_cols: List of numerical column names
            cat_cols: List of categorical column names.
        return:
            vif_high: list of features having high VIF>10
        """
        y, X = dmatrices(targetcol+'~'+'+'.join(num_cols+cat_cols),
                         data=data, return_type='dataframe')
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(
            X.values, i) for i in range(X.shape[1])]
        vif["features"] = X.columns
        vif_high = list(vif[vif['VIF Factor'] > 10]['features'])
        if 'Intercept' in vif_high:
            vif_high.remove('Intercept')
        vif_high = list(set([re.sub(r'\[.*', '', s) for s in vif_high]))
        return vif_high

    def removal_on_vif(targetcol: str = None, data: pd.DataFrame = None, num_cols: list = None, cat_cols: list = None):
        """
        Remove all the features having high VIF.
        args:
            targetcol: target column name
            data: whole cleaned dataframe
            num_cols: List of numerical column names
            cat_cols: List of categorical column names
        return: 
            num_cols: list containing numerical column names after removal based on VIF
            cat_cols: list containing categorical column names after removal based on VIF
            feat_removal: features removed on basis of VIF>10
        """
        vif_first = LinearModel.vif(targetcol, data, num_cols, cat_cols)
        print("\n VIF first run on all the columns:\n", len(vif_first))
        feat_removal = []
        for i in vif_first:
            cols = list(set(num_cols).difference(vif_first))
            cols.append(i)
            cols1 = list(set(cat_cols).difference(vif_first))
            ans = LinearModel.vif(targetcol, data[cols+cols1+[targetcol]],
                                  cols, list(set(cat_cols).difference(vif_first)))
            for j in ans:
                if j in vif_first:
                    feat_removal.append(j)
        feat_removal = sorted(list(set([re.sub(r'\[.*', '', s)
                                        for s in feat_removal])))
        num_cols = list(set(num_cols).difference(feat_removal))
        cat_cols = list(set(cat_cols).difference(feat_removal))
        print(f"\n Columns Removed on basis of VIF:\n {feat_removal}")
        return sorted(num_cols), sorted(cat_cols), feat_removal

    def removal_on_pvalue(targetcol: str = None, num_cols: list = None, cat_cols: list = None, data: pd.DataFrame = None):
        """
            Fit model and remove the columns based on pvalue test on each feature.
        args:
            targetcol: target column name
            data: whole cleaned dataframe
            num_cols: List of numerical column names
            cat_cols: List of categorical column names
        """
        model = smf.ols(
            targetcol+'~'+'+'.join(num_cols+cat_cols), data=data).fit()
        pvalue_removal = sorted(
            list(model.pvalues[model.pvalues > 0.05].keys()))
        if 'Intercept' in pvalue_removal:
            pvalue_removal.remove("Intercept")
        #pvalue_removal = list(set([re.sub(r'\[.*', '', s) for s in pvalue_removal]))
        num_cols = list(set(num_cols).difference(pvalue_removal))
        cat_cols = list(set(cat_cols).difference(pvalue_removal))
        print("\n Columns removed on basis of Pvalue:\n", pvalue_removal)
        return sorted(num_cols), sorted(cat_cols), pvalue_removal

    def anovatyp1_removal(targetcol: str = None, num_cols: list = None, cat_cols: list = None, data: pd.DataFrame = None):
        """
            Fit anova test and remove the columns failing the anova type 1 test.
        args:
            targetcol: target column name
            data: whole cleaned dataframe
            num_cols: List of numerical column names
            cat_cols: List of categorical column names
            anovatyp1_removal: columns removed on anova type 1 test
            """
        model = smf.ols(
            targetcol+'~'+'+'.join(num_cols+cat_cols), data=data).fit()
        anova_typ1 = sm.stats.anova_lm(model, typ=1)
        m = anova_typ1['PR(>F)']
        anovatyp1_removal = sorted(list(m[m > 0.05].index))
        print("\n Columns removed on basis of anova typ1:\n", anovatyp1_removal)
        num_cols = list(set(num_cols).difference(anovatyp1_removal))
        cat_cols = list(set(cat_cols).difference(anovatyp1_removal))
        return sorted(num_cols), sorted(cat_cols), anovatyp1_removal

    def finalmodel(targetcol: str = None, num_cols: list = None, cat_cols: list = None, data: pd.DataFrame = None):
        """
        Fit final OLS model on data and also get explainability based on tstat values.
        args:
            targetcol: target column name
            data: whole cleaned dataframe
            num_cols: List of numerical column names
            cat_cols: List of categorical column names
        """
        print(f"\n Final Model:{targetcol}+'~'+{'+'.join(num_cols+cat_cols)}")
        model = smf.ols(
            targetcol+'~'+'+'.join(num_cols+cat_cols), data=data).fit()
        explanation = np.abs(model.tvalues).sort_values(ascending=False)
        explanation = explanation[explanation.index != "Intercept"]
        #sns.regplot(x = data[targetcol], y = model.fittedvalues)
        plt.xlabel("True Price values")
        plt.ylabel("Predicted Price values")
        return model, explanation, num_cols, cat_cols

    def adjusted_r2(predictions: list = None, data: str = pd.DataFrame, targetcol: str = None):
        """
        Calculate adjusted R2.
        args:
            predictions: list of all the fitted model prediction values.
            data: DataFrame of initial dataset containing all columns.
            targetcol: target column name.
        """
        try:
            r2 = r2_score(predictions, data[targetcol])
            adjr2 = 1-(1-r2)*(len(data)-1)/(len(data)-len(data.columns)-1)
            return adjr2
        except:
            print("Unable to calculate adjusted R2, please check predictions list.")

    def bootstrap_confidence(n_iterations, targetcol, num_cols, cat_cols, data):
        """
        get confidence score of the model using bootstrapping.
        args:
            targetcol: target column name
            data: whole cleaned dataframe
            num_cols: List of numerical column names
            cat_cols: List of categorical column names
            n_iterations: number of iteration to run bootstrapping
        return:
            Confidence interval: Lower and upper quantile.
        """
        try:
            stats = list()
            for i in range(n_iterations):
                # prepare train and test sets
                train, test, y_train, y_test = train_test_split(
                    data[num_cols+cat_cols], data[targetcol], test_size=0.2, random_state=0)
                train[targetcol] = y_train
                test[targetcol] = y_test
                # fit model
                model = smf.ols(targetcol+'~'+'+'.join(num_cols +
                                cat_cols), data=train).fit()
                # evaluate model
                predictions = model.predict(test[num_cols+cat_cols])
                score = adjusted_r2(predictions, test, targetcol)
                if i % 100 == 0:
                    print(f"Adj R2 for iter {i}: {np.round(score,3)*100}%")
                stats.append(score)
            alpha = 0.95
            p = ((1.0-alpha)/2.0) * 100
            lower = max(0.0, np.percentile(stats, p))
            p = (alpha+((1.0-alpha)/2.0)) * 100
            upper = min(1.0, np.percentile(stats, p))
            print('%.1f%% confidence interval %.1f%% and %.1f%%' %
                  (alpha*100, lower*100, upper*100))
            return lower*100, upper*100
        except:
            print("\n Not enough resampled data")
            return 0, 0

    def ci_pi(targetcol, num_cols, cat_cols, data, mse, model):
        """
        Calculate Confidence Interval and Prediction Interval for a given Sample
        args:
            targetcol: target column name.
            num_cols: list of numerical columns.
            cat_cols: list of categorical columns.
            data: DataFrame of whole initial dataset. 
        return:
            2 tuples Confidence Interval and Prediction Interval
        """
        try:
            y, X = dmatrices(targetcol+'~'+'+'.join(num_cols+cat_cols),
                             data=data, return_type='dataframe')
            stderr = np.sqrt(
                mse*np.dot(X.iloc[0].values, np.dot(np.linalg.inv(np.dot(X.T, X)), X.iloc[0].values.T)))
            t_conf = st.t.ppf(0.025, len(X)-len(X.columns))
            y_hat = model.predict(pd.DataFrame(
                data[num_cols+cat_cols].iloc[0:3]))[1]
            conf_interval = y_hat-(np.abs(t_conf)*stderr), y_hat + \
                (np.abs(t_conf)*stderr)
            pred_interval = y_hat-(np.abs(t_conf)*np.sqrt((stderr**2)+mse)
                                   ), y_hat+(np.abs(t_conf)*np.sqrt((stderr**2)+mse))
            return conf_interval, pred_interval
        except:
            return 0, 0

    def influential_points(targetcol: str = None, num_cols: list = None, cat_cols: list = None, data: pd.DataFrame = None):
        """
        Calculate cooks distance and studentized residual test.
        Common elements in both cook's and studentized are removed.
        args:
            targetcol: target column name
            data: whole cleaned dataframe
            num_cols: List of numerical column names
            cat_cols: List of categorical column names
            n_iterations: number of iteration to run bootstrapping
        return:
            Confidence interval: Lower and upper quantile.
        """
        n = len(data)
        p = len(num_cols+cat_cols)+1

        model = smf.ols(
            targetcol+'~'+'+'.join(num_cols+cat_cols), data=data).fit()
        infl = model.get_influence()
        inflsum = infl.summary_frame()
        reg_cook = inflsum.cooks_d
        atyp_cook = np.abs(reg_cook) >= 4/len(data)
        cook_infl_points = list(data.index[atyp_cook])

        seuil_stud = scipy.stats.t.ppf(0.975, df=n-p-1)
        # detection - absolute value > threshold
        reg_studs = infl.resid_studentized_external
        atyp_stud = np.abs(reg_studs) > seuil_stud
        # which ones?
        student_inflpoints = list(data.index[atyp_stud])
        outliers = [i for i in student_inflpoints if i in cook_infl_points]
        print("\n Index of influential points detected by Cooks and Studentized:", outliers)
        return outliers

    def bp_test_fn(model):
        """
        args:
            model: fitted model
        return:
            bresch pegan test results
        """
        bp_test = het_breuschpagan(model.resid, model.model.exog)
        labels = ['LM Statistic', 'LM-Test p-value']
        bp_result = dict(zip(labels, bp_test))
        return bp_result

    # def normality_qqplot(model):
    #     sm.qqplot(model.resid)
    #     sns.set_style('whitegrid')
    #     plt.title('QQ plot', fontsize=18, fontweight="bold")
    #     plt.xlabel('Theoretical Quantiles', fontsize=14, fontweight="bold")
    #     plt.ylabel('Sample Quantiles', fontsize=14, fontweight="bold")
    #     plt.show()
    #     return
