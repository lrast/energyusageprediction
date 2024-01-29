import polars as pl


class History_Holder(object):
    """History_Holder: object holding data to be updated online
     as new data comes in"""
    def __init__(self, data):
        super(History_Holder, self).__init__()
        self.data = data

    def update_data(self, new_data):
        self.data = pl.concat([self.data, new_data], how='vertical')

