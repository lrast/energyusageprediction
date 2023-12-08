# preprocessing the targets before training
import polars as pl
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, PowerTransformer

from EnefitTools.features.utilities import PolarsInPlaceTransforms


LogTransformer = PolarsInPlaceTransforms(
            transformers=[('nlin',  
                           FunctionTransformer(func=np.log1p, inverse_func=np.expm1),
                          ["target"]
                           )]
        )


class IdNormalizer(BaseEstimator, TransformerMixin):
    """ IdNormalizer: normalizes the data to the maximum in a given unit id
        To do: update for unit ids that have never been seen.
    """
    def __init__(self):
        super(IdNormalizer, self).__init__()
        self.max_values = None

    def fit(self, data, y=None):
        """ Preprocess the training features, to find unit-wise max output"""
        max_values = data.group_by(
                            ['prediction_unit_id']
                         ).agg(max_value=pl.col('target').max())
        self.max_values = max_values
        return self

    def transform(self, targets):
        # to improve: better filling of unobserved units
        targets = targets.join(
                               self.max_values, 
                               on=['prediction_unit_id'],
                               how='left'
                        ).fill_nan(
                                1.0
                        ).with_columns(
                                target=pl.col('target')/pl.col('max_value')
                        ).drop(['max_value'])
        return targets

    def inverse_transform(self, predictions):
        """ undo the maximum normalization """
        predictions = predictions.join(
                                    self.max_values,
                                    on=['prediction_unit_id'],
                                    how='left'
                                ).with_columns(
                                    target=pl.col('target')*pl.col('max_value')
                                ).drop(['max_value', 'prediction_unit_id'])
        return predictions
