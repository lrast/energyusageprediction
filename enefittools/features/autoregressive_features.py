import datetime
import polars as pl

from sklearn.base import BaseEstimator, TransformerMixin


class Delayed_Features(BaseEstimator, TransformerMixin):
    """Transformer for adding time-shifted historical data"""
    def __init__(self, to_delay, history):
        self.to_delay = to_delay
        self.history = history

    def fit(self, full_data, y=None):
        self.column_names_ = ['2d_ago', '7d_ago']
        return self

    def transform(self, full_data, drop_nulls=False):
        """Add time-shifted historical data"""
        historical = self.history.data

        out = full_data.with_columns( 
                        (pl.col('prediction_datetime') - datetime.timedelta(days=2)).alias('2d_ago'),
                        (pl.col('prediction_datetime') - datetime.timedelta(days=7)).alias('7d_ago')
                    ).join(historical.rename({self.to_delay: self.to_delay+'_2d_ago'}),
                           left_on=['prediction_unit_id', 'is_consumption', '2d_ago'],
                           right_on=['prediction_unit_id', 'is_consumption', 'prediction_datetime'],
                           how='left'
                    ).join(historical.rename({self.to_delay: self.to_delay+'_7d_ago'}),
                           left_on=['prediction_unit_id', 'is_consumption', '7d_ago'],
                           right_on=['prediction_unit_id', 'is_consumption', 'prediction_datetime'],
                           how='left'
                    ).drop('2d_ago'
                    )

        if drop_nulls:
            return out.drop_nulls()
        else:
            return out.fill_null(0)

    def fit_transform(self, X, y=None):
        """ Callled during fitting, we want to drop nulls. """
        self.fit(X)
        return self.transform(X, drop_nulls=True)
