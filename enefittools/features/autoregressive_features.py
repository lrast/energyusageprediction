import datetime
import polars as pl

from sklearn.base import BaseEstimator, TransformerMixin


class Delayed_Features(BaseEstimator, TransformerMixin):
    """Transformer for adding time-shifted historical data"""
    def __init__(self, to_delay):
        self.to_delay = to_delay

    def fit(self, data_holder, y=None):
        self.column_names_ = ['2d_ago', '7d_ago']
        return self

    def transform(self, data_holder):
        """Add time-shifted historical data"""
        full_data = data_holder.features
        historical = data_holder.target_historical

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

        # drop nulls for training
        if data_holder.mode == 'train':
            out = out.drop_nulls()
        else:
            out = out.fill_null(0)

        data_holder.features = out
        return data_holder
