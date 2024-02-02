import polars as pl

import statsmodels.formula.api as smf
from sklearn.base import BaseEstimator, RegressorMixin


class SM_Regression(BaseEstimator, RegressorMixin):
    """ Wrapper for statsmodels formula OLS
    """
    def __init__(self, formula):
        self.formula = formula

        self.model = None
        self.is_fit = False

    def fit(self, data_holder, overwrite=False):
        if self.is_fit and not overwrite:
            # protection against costly re-fitting
            raise Exception('Already fit')

        self.model = smf.ols(self.formula, data=data_holder.features.to_pandas())
        self.model = self.model.fit()
        
        self.model.remove_data()
        self.is_fit = True
        self.is_fit_ = True
        return self

    def predict(self, data_holder):
        predictions = self.model.predict(data_holder.features.to_pandas())

        data_holder.features = data_holder.features.select(
                        data_holder.target.columns
                  ).with_columns(
                        prediction=pl.lit(pl.from_pandas(predictions))
                  )

        return data_holder


class Multiple_Regressions(object):
    """docstring for Multiple_regressions"""
    def __init__(self, models):
        super(Multiple_Regressions, self).__init__()
        self.models = models

    def fit(self, data_holder):
        all_features = data_holder.features

        f1 = all_features.filter(pl.col('is_business') == True)
        f0 = all_features.filter(pl.col('is_business') == False)

        data_holder.features = f1
        self.models[1].fit(data_holder)
        results1 = data_holder.features

        data_holder.features = f0
        self.models[0].fit(data_holder)
        results0 = data_holder.features

        data_holder.features = pl.concat([results0, results1], how='vertical')

        self.is_fit_ =True
        return self

    def predict(self, data_holder):
        all_features = data_holder.features

        f0 = all_features.filter(pl.col('is_business') == False)
        f1 = all_features.filter(pl.col('is_business') == True)

        data_holder.features = f0
        pred0 = self.models[0].predict(data_holder).features

        data_holder.features = f1
        pred1 = self.models[0].predict(data_holder).features

        data_holder.features = pl.concat([pred0, pred1], how='vertical')

        return data_holder
