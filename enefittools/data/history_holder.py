import polars as pl


class Target_History(object):
    """History_Holder: object holding data to be updated online
     as new data comes in"""
    def __init__(self, data, normalize=False):
        super(Target_History, self).__init__()
        self.keys = ['prediction_datetime', 'prediction_unit_id', 'is_consumption']
        self.normalize = normalize
        self.data = self.preprocess(data)

    def preprocess(self, data):
        if self.normalize:
            data = data.with_columns(pl.col('target') / pl.col('installed_capacity'))

        return data.select(self.keys + ['target'])

    def update_data(self, new_data):
        new_data = self.preprocess(new_data)

        latest_seen = self.data['prediction_datetime'].max()
        unseen_data = new_data.filter(pl.col('prediction_datetime') > latest_seen)

        self.data = pl.concat([self.data, unseen_data], how='vertical').unique()
