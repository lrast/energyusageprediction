import polars as pl

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor


class RF_Residual(BaseEstimator, RegressorMixin):
    """RF_Residuals: trains a random forest model on the residuals"""
    def __init__(self, **model_kwargs):
        super(RF_Residual, self).__init__()
        self.model = RandomForestRegressor(**model_kwargs)
        self.rf_features = []

    def fit(self, data_holder, y=None):
        column_names = ['residual'] + self.get_column_names(data_holder)

        train_data = data_holder.features.with_columns(
                                        residual=(pl.col('target') - pl.col('prediction'))
                                    ).select(column_names)

        fit_targets = train_data['residual'].to_numpy()
        fit_inputs = train_data.drop('residual').to_numpy()

        self.model.fit(fit_inputs, fit_targets)
        return self

    def predict(self, data_holder):
        column_names = self.get_column_names(data_holder)

        previous_predictions = data_holder.features['prediction']
        current_features = data_holder.features.select(column_names)

        my_predictions = self.model.predict(current_features.to_numpy())

        out = data_holder.features.select(data_holder.target.columns
                                 ).with_columns(
                                    prediction=pl.lit(my_predictions) + \
                                    pl.lit(previous_predictions)
                                  )
        data_holder.features = out
        return data_holder

    def get_column_names(self, data_holder):
        return ['eic_count', 'installed_capacity'] + data_holder.features.columns[-15:] 


