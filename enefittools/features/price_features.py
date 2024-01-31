# adds price data to the features
from sklearn.base import BaseEstimator, TransformerMixin


class Price_Features(BaseEstimator, TransformerMixin):
    """Transformer that adds Price Features to the data"""
    def __init__(self):
        super(Price_Features, self).__init__()

    def fit(self, data_holder, y=None):
        self.is_fit_ = True
        return self

    def transform(self, data_holder, y=None):
        features = data_holder.features

        features = features.join(
                            data_holder.prices_gas.select('forecast_date',
                                                          'lowest_price_per_mwh',
                                                          'highest_price_per_mwh'),
                            left_on='date',
                            right_on='forecast_date',
                            how='left'
                           ).join(
                            data_holder.prices_electricity.select('forecast_datetime',
                                                                  'euros_per_mwh'),
                            left_on='prediction_datetime',
                            right_on='forecast_datetime',
                            how='left'
                            ).fill_null(0
                            ).rename({'lowest_price_per_mwh': 'price_gas_low',
                                      'highest_price_per_mwh': 'price_gas_high',
                                      'euros_per_mwh': 'price_electricity'
                                      })


        data_holder.features = features

        return data_holder
