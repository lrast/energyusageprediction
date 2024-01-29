import pandas as pd
import polars as pl

from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess


class Datetime_Features(BaseEstimator, TransformerMixin):
    """Transformer for creating datetime features"""
    def __init__(self):
        pass
    
    def fit(self, target_data, y=None):
        dates = pd.date_range(target_data['prediction_datetime'].dt.date().min(), 
                              target_data['prediction_datetime'].dt.date().max())
        fourier = CalendarFourier(freq="A", order=6) 
        date_process = DeterministicProcess(
                                dates,
                                constant=False,
                                order=1,
                                additional_terms=[fourier],
                                drop=True)
        self.date_process_ = date_process

        _, date_names = self.make_date_features(dates)
        
        self.column_names_ = ['weekday', 'hour_of_day'] + date_names
        return self

    def make_date_features(self, target_dates):
        raw_features = self.date_process_.range(target_dates.min(), target_dates.max())
        raw_features = pl.from_pandas(raw_features.reset_index())

        name_dict = {old_name: f'{old_name[0:3]}_{old_name[4]}'
                     for old_name in raw_features.columns[-12:]}

        raw_features = raw_features.rename({'index': 'date'}
                                  ).rename(name_dict)

        return raw_features, ['trend'] + list(name_dict.values())

    def transform(self, full_data):
        date_features, _ = self.make_date_features(full_data['date'].unique())

        full_data = full_data.with_columns( 
                                pl.col('prediction_datetime').dt.weekday().alias('weekday'),
                                pl.col('prediction_datetime').dt.hour().alias('hour_of_day')
                            ).join(
                                    date_features.with_columns(pl.col('date').dt.date()),
                                    on='date'
                            )
        return full_data
