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

    def transform(self, data_holder):
        X = data_holder.features
        if self.mode == 'fwd':
            if data_holder.mode == 'train':
                outs = X.with_columns(pl.col('target') / pl.col('installed_capacity'))
            else:
                outs = X
        if self.mode == 'inv':
            outs = X.with_columns(pl.col('prediction') * pl.col('installed_capacity'))

        data_holder.features = outs
        return data_holder

    def predict(self, data_holder):
        if self.mode == 'inv':
            data_holder.features = data_holder.features.with_columns(
                                            pl.col('prediction') * pl.col('installed_capacity')
                                            )
            return data_holder
