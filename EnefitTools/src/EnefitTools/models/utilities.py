import polars as pl


def make_wrapped_model(modelClass):
    """ Wraps model class to unpackage and repackage inputs and results 
        into dataframes when fitting and predicting
    """

    class WrappedModel(modelClass):
        def __init__(self):
            super(WrappedModel, self).__init__()

        def predict(self, features):
            row_ids = features['row_id']
            unit_ids = features['prediction_unit_id']
            outputs = super(WrappedModel, self).predict(
                                features.drop(['row_id', 'prediction_unit_id'])
                            )

            return pl.DataFrame({'row_id': row_ids, 'prediction_unit_id': unit_ids,
                                 'target': outputs})

        def fit(self, X, y):
            trainSet = X.join(y[['row_id', 'target']], on='row_id', how='inner')
            targets = trainSet['target']
            features = trainSet.drop(['target', 'row_id', 'prediction_unit_id'])

            super(WrappedModel, self).fit(features, targets)

    return WrappedModel()
