import datetime
import polars as pl
import pandas as pd


def format_dfs(target=None, revealed_targets=None, client=None,
               weather_historical=None, weather_forecast=None,
               electricity_prices=None, gas_prices=None, 
               sample_prediction=None, solar=None):
    """ dataframe formatting for training and online use 
        
        - encodes as polars dataframes
        - sets proper types
        - filter out two weather stations that don't make sense
    """

    column_types = {'county': pl.Int8, 'product_type': pl.Int8,
                    'is_business': pl.Boolean, 'is_consumption': pl.Boolean,
                    'row_id': pl.Int64, 'prediction_unit_id': pl.Int8,

                    'prediction_datetime': pl.Datetime('ms'),
                    'origin_datetime': pl.Datetime('ms'),
                    'forecast_datetime': pl.Datetime('ms'),
                    'datetime': pl.Datetime('ms')
                    }

    if target is not None:
        # target formatting
        target.dropna(axis='index', inplace=True)
        target = target.rename(columns={'datetime': 'prediction_datetime'})
        target['prediction_datetime'] = pd.to_datetime(target['prediction_datetime'])
        target = pl.from_pandas(target,
                                schema_overrides=column_types
                                )
        target = target.with_columns(
                    (pl.col('prediction_datetime').dt.date() +
                        datetime.timedelta(days=-2)).alias('date_when_predicting'),
                  )

    if revealed_targets is not None:
        revealed_targets = pl.from_pandas(revealed_targets,
                                          schema_overrides=column_types)

    if client is not None:
        # client formatting
        client['date'] = pd.to_datetime(client['date']).dt.date
        client = pl.from_pandas(client, schema_overrides=column_types)
        client = client.with_columns()

    if weather_historical is not None:
        weather_historical['datetime'] = pd.to_datetime(weather_historical['datetime'])
        weather_historical = pl.from_pandas(weather_historical,
                                            schema_overrides=column_types)
        weather_historical = weather_historical.filter(
                                ((pl.col('latitude') != 57.6) | (pl.col('longitude') != 23.2)),
                                ((pl.col('latitude') != 57.6) | (pl.col('longitude') != 24.2)),
                               )

    if weather_forecast is not None:
        weather_forecast['origin_datetime'] = pd.to_datetime(weather_forecast['origin_datetime'])
        weather_forecast['forecast_datetime'] = pd.to_datetime(weather_forecast['forecast_datetime'])
        weather_forecast = pl.from_pandas(weather_forecast,
                                          schema_overrides=column_types)
        weather_forecast = weather_forecast.filter(
                                ((pl.col('latitude') != 57.6) | (pl.col('longitude') != 23.2)),
                                ((pl.col('latitude') != 57.6) | (pl.col('longitude') != 24.2)),
                               )

    if electricity_prices is not None:
        electricity_prices = pl.from_pandas(electricity_prices, schema_overrides=column_types)

    if gas_prices is not None:
        gas_prices = pl.from_pandas(gas_prices, schema_overrides=column_types)

    if sample_prediction is not None:
        pass

    if solar is not None:
        solar['datetime'] = pd.to_datetime(solar['datetime'])
        solar = pl.from_pandas(solar, schema_overrides=column_types)
        if 'latitude' in solar and 'longitude' in solar:
            solar = solar.filter(
                        ((pl.col('latitude') != 57.6) | (pl.col('longitude') != 23.2)),
                        ((pl.col('latitude') != 57.6) | (pl.col('longitude') != 24.2)),
                       )

    return tuple(filter(lambda x: x is not None,
                        (target, revealed_targets, client, 
                         weather_historical, weather_forecast,
                         electricity_prices, gas_prices, sample_prediction, solar
                         )))


def split_production_consumption(features, targets):
    """ splits the features and targets into production and consumption set
    """
    production_features = features.filter(
                                        pl.col('is_consumption') == 0
                                 ).drop('is_consumption')
    production_targets = targets.filter(
                                        pl.col('is_consumption') == 0
                               ).drop('is_consumption')

    consumption_features = features.filter(
                                        pl.col('is_consumption') == 1
                                  ).drop('is_consumption')
    consumption_targets = targets.filter(
                                        pl.col('is_consumption') == 1
                                ).drop('is_consumption')

    return (production_features, production_targets,
            consumption_features, consumption_targets)


def localToUTC(df, timeInd, location):
    """Converts local time to UTC, and adds missing hours"""
    localTime = pd.to_datetime(df[timeInd]).dt.tz_localize(location, ambiguous=True)
    df[timeInd] = localTime.dt.tz_convert('UTC')

    allTimes = pd.date_range(df[timeInd].min(), df[timeInd].max(), freq='H')
    missingTimes = allTimes.difference(df[timeInd])

    for missing in missingTimes:
        prev = missing + datetime.timedelta(hours=-1)
        fillData = df[df[timeInd] == prev].iloc[-1].to_dict()
        fillData[timeInd] = missing

        df.loc[len(df)] = fillData

    return df
