# add solar data to the features
from sklearn.base import BaseEstimator, TransformerMixin


class Solar_Features(BaseEstimator, TransformerMixin):
    """Transformer that adds Solar Features to the data"""
    def __init__(self, additional_joins=[]):
        super(Solar_Features, self).__init__()
        self.additional_joins = additional_joins

    def fit(self, data_holder, y=None):
        return self

    def transform(self, data_holder, y=None):
        left_keys = ['prediction_datetime'] + self.additional_joins
        right_keys = ['datetime'] + self.additional_joins

        outs = data_holder.features.join(data_holder.solar,
                                         left_on=left_keys, right_on=right_keys,
                                         how='left')
        data_holder.features = outs
