import polars as pl

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer


class PolarsInPlaceTransforms(BaseEstimator, TransformerMixin):
    """
        PolarsInPlaceTransforms: in place operations on polars columns
    """
    def __init__(self, transformers):
        super(PolarsInPlaceTransforms, self).__init__()
        self.PandasTransform = ColumnTransformer(transformers)
        self.columns = sum(map(lambda x: x[2], transformers), [])
        self.transformers = transformers

    def fit(self, X, y=None):
        if y:
            y = y.to_pandas()
        self.PandasTransform.fit(X.to_pandas(), y)
        return self

    def transform(self, X, y=None):
        outputs = X.clone()
        results = pl.from_numpy(self.PandasTransform.transform(X.to_pandas()))

        for i, name in enumerate(self.columns):
            outputs = outputs.with_columns(results[:, 0].alias(name))

        return outputs

    def inverse_transform(self, X, y=None):
        for (name, transform, columnNames) in self.transformers:
            dataFrameChanges = map(
                lambda colName: transform.inverse_transform(pl.col(colName)),
                columnNames)
            X = X.with_columns(dataFrameChanges)
        return X
