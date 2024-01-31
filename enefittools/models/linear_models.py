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
