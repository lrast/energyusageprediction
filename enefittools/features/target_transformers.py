# methods for target transformation

import polars as pl

from sklearn.base import BaseEstimator, TransformerMixin


class Normalize_Target(BaseEstimator, TransformerMixin):
    """Normalize_Target: basic transformation for normalizing and
        un-normalizing the targets by installed capacity
    """
    def __init__(self, mode='fwd'):
        super(Normalize_Target, self).__init__()
        self.mode = mode

    def fit(self, *args):
        self.is_fit_ = True
        return self

    def transform(self, X):
        if self.mode == 'fwd':
            if 'target' in X.columns:
                return X.with_columns(pl.col('target') / pl.col('installed_capacity'))
            else:
                return X
        if self.mode == 'inv':
            return X.with_columns(pl.col('prediction') * pl.col('installed_capacity'))

    def predict(self, X):
        if self.mode == 'inv':
            return X.with_columns(pl.col('prediction') * pl.col('installed_capacity'))
