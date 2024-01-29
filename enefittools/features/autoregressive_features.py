import datetime
import polars as pl

from sklearn.base import BaseEstimator, TransformerMixin
import sklearn
sklearn.set_config(enable_metadata_routing=True)


class Delayed_Features(BaseEstimator, TransformerMixin):
    """Transformer for adding time-shifted historical data"""
    def __init__(self, to_delay):
        self.to_delay = to_delay
        self.set_transform_request(historical=True)

    def fit(self, full_data, y=None):
        self.column_names_ = ['2d_ago', '7d_ago']
        return self

    def transform(self, full_data, historical=None):
        """Add time-shifted historical data"""
        provided_data = historical
        
        if historical is None:
            historical = full_data

        historical = historical[[self.to_delay, 'prediction_unit_id',
                                'is_consumption', 'prediction_datetime']]

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

        if provided_data is None:
            return out.drop_nulls()
        else:
            return out.fill_null(0)
