# linear models for the forecasting component

import polars as pl

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.linear_model import LinearRegression

from enefittools.models.utilities import make_wrapped_model


class TimeseriesFeatures(BaseEstimator, TransformerMixin):
    """ TimeseriesFeatures: use linear regression on different subsets
        of the data to make features for future regressors.

        For now, the regressor groups are based on the client information
    """
    def __init__(self):
        super(TimeseriesFeatures, self).__init__()
        self.models = {
            'county': {i: make_wrapped_model(LinearRegression) for i in range(16)},
            'is_business': {i: make_wrapped_model(LinearRegression) for i in range(2)},
            'product_type': {i: make_wrapped_model(LinearRegression) for i in range(4)}
            }

    def fit(self, X, y):
        # 1) normalize the feature by the unit max
        unitMaxes = y.group_by('prediction_unit_id').agg(
                            pl.col('target').max().alias('unit_max'))
        y = y.join(unitMaxes, on='prediction_unit_id'
            ).with_columns(target=(pl.col('target')/pl.col('unit_max'))
            ).drop('unit_max')

        # 2) fit each model on the corresponing data slice
        for columnName, possible in self.models.items():
            for value, model in possible.items():
                features_current = X.filter(pl.col(columnName) == value)
                targets_current = y.filter(pl.col(columnName) == value)

                model.fit(features_current, targets_current)

        return self

    def transform(self, X, y=None):
        runningResults = None
        for columnName, possible in self.models.items():
            for value, model in possible.items():
                predictions_current = model.predict(X).rename(
                                                 {'target': columnName+str(value)}
                                                )

                if runningResults is None:
                    runningResults = predictions_current
                else:
                    runningResults = runningResults.join(predictions_current,
                                                         on=['row_id', 'prediction_unit_id'])

        return runningResults


class ParallelLinearModels(BaseEstimator, RegressorMixin):
    """ParallelLinearModels: split the input data based on some features and
       apply a linear regression to each subset
    """
    def __init__(self):
        super(ParallelLinearModels, self).__init__()
        feature = ('prediction_unit_id', range(69))
        self.models = {key: make_wrapped_model(LinearRegression) 
                       for key in feature[1]}
        self.col_name = feature[0]

    def fit(self, X, y):
        for key, model in self.models.items():
            model.fit(X.filter(pl.col(self.col_name) == key),
                      y.filter(pl.col(self.col_name) == key))
        return self

    def predict(self, X):
        predictions = []
        for key, model in self.models.items():
            if X.filter(pl.col(self.col_name) == key).shape[0] != 0:
                predictions.append(
                            model.predict(X.filter(pl.col(self.col_name) == key))
                        )

        return pl.concat(predictions, how='vertical')
