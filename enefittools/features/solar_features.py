# add solar data to the features
from sklearn.base import BaseEstimator, TransformerMixin


class Solar_Features(BaseEstimator, TransformerMixin):
    """Transformer that adds Solar Features to the data"""
    def __init__(self, dataset, additional_joins=[]):
        super(Solar_Features, self).__init__()
        self.dataset = dataset
        self.additional_joins = additional_joins

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        left_keys = ['prediction_datetime'] + self.additional_joins
        right_keys = ['datetime'] + self.additional_joins

        return X.join(self.dataset,
                      left_on=left_keys, right_on=right_keys,
                      how='left')
