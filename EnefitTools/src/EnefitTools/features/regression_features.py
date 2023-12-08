# make deterministic time dependent features
import pandas as pd
import polars as pl
import numpy as np

from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess


def make_date_process(train):
    dates = pd.date_range(train['prediction_datetime'].dt.date().min(), 
                          train['prediction_datetime'].dt.date().max())
    fourier = CalendarFourier(freq="A", order=6) 
    date_process = DeterministicProcess(
                            dates,
                            constant=True,
                            order=0,
                            seasonal=True,
                            period=7,
                            additional_terms=[fourier],
                            drop=True
                          )
    return date_process


def make_regression_features(train, clients, date_process):
    """ input features for the regression"""
    features = train.drop('target').join(
                    clients, 
                    left_on=['county', 'is_business', 'product_type', 'date_when_predicting'],
                    right_on=['county', 'is_business', 'product_type', 'date'],
                    how='inner'
                    )

    # add date and time features
    features = features.with_columns( 
                        pl.col('prediction_datetime').dt.date().alias('prediction_date'),
                        pl.col('prediction_datetime').dt.time().alias('prediction_time')
                        )

    # features from the date process
    date_features = date_process.range(
                                       features['prediction_date'].min(),
                                       features['prediction_date'].max()
                                      ).reset_index()
    date_features = pl.from_pandas(date_features).with_columns(pl.col('index').dt.date())
    features = features.join(date_features,
                             left_on=['prediction_date'],
                             right_on=['index'],
                             how='left')

    # time features:
    # to add: local holidays
    features = features.to_dummies('prediction_time') 
    features.drop('prediction_time_10:00:00')

    # Other covariates
    features = features.with_columns(np.log1p(pl.col('installed_capacity')))
    
    # some of these features should be included in the future
    columns_to_drop = ['eic_count',
                       'county', 'is_business', 'product_type',
                       'data_block_id', 'data_block_id_right',
                       'prediction_date', 'prediction_datetime', 'date_when_predicting'
                       ]
    features = features.drop(columns_to_drop)
    
    return features
