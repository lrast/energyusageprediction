# models that aid in chaining together multiple predictors
import polars as pl

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin


class Predictions_to_Features(BaseEstimator, TransformerMixin):
    """Predictions_to_Features takes a regressor or classifier and wraps it
        into a transformer
    """
    def __init__(self, model):
        super(Predictions_to_Features, self).__init__()
        self.model = model

    def fit(self, X):
        self.model = self.model.fit(X)
        self.is_fit_ = True
        return self

    def transform(self, X):
        return self.model.predict(X)


class Learn_On_Residuals(BaseEstimator, RegressorMixin):
    """Learn_On_Residuals allows boosting of the previous model in the pipeline
        by the next model, by learning on the residuals and predicting with the
        sums of the predictors
    """
    def __init__(self, model, input_targets='target', input_predictions='prediction',
                 output_targets='targets'):
        super(Learn_On_Residuals, self).__init__()
        self.model = model
        self.input_targets = input_targets
        self.input_predictions = input_predictions
        self.output_targets = output_targets

    def fit(self, data):
        """Learn this model on the residuals from the previous predictions"""
        to_fit = data.with_columns(
                        (pl.col(self.input_targets) - pl.col(self.input_predictions)
                         ).alias('__residuals__')
                    ).drop(
                        self.input_targets, self.input_predictions
                    ).rename({'__residuals__': self.output_targets})

        self.model = self.model.fit(to_fit)
        self.is_fit_ = True
        return self

    def predict(self, X):
        """Return the sum of these predictions and the previous predictions"""
        prediction_features = X.drop(self.input_targets, self.input_predictions)

        predictions = self.model.predict(prediction_features)

        return predictions.with_columns(pl.col(self.output_targets) +
                                        pl.lit(X[self.input_predictions])
                                        )
