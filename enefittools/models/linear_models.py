import polars as pl

import statsmodels.formula.api as smf
from sklearn.base import BaseEstimator, RegressorMixin


class SM_Regression(BaseEstimator, RegressorMixin):
    """ Wrapper for statsmodels formula OLS
    """
    def __init__(self, formula, to_drop):
        self.formula = formula
        self.to_drop = to_drop

        self.model = None
        self.is_fit = False

    def fit(self, data, overwrite=False):
        if self.is_fit and not overwrite:
            # protection against costly re-fitting
            raise Exception('Already fit')

        self.model = smf.ols(self.formula, data=data.to_pandas())
        self.model = self.model.fit()
        
        self.model.remove_data()
        self.is_fit = True
        self.is_fit_ = True
        return self
    
    def predict(self, X):
        predictions = self.model.predict(X.to_pandas())

        outputs = X.drop(
                        self.to_drop
                  ).with_columns(
                        prediction=pl.lit(pl.from_pandas(predictions))
                  )

        return outputs

    def residuals(self, data, target_col='target'):
        predictions = pl.from_pandas(self.predict(data))

        return data.with_columns(
                        prediction=pl.lit(predictions)
                  ).with_columns(
                        residual=pl.col(target_col)-pl.col('prediction')
                  )
