import datetime
import polars as pl
import pandas as pd


def format_dfs(test, client):
    """ dataframe formatting for training and online use """
    # test formatting
    test.dropna(axis='index', inplace=True)
    test = test.rename(columns={'datetime': 'prediction_datetime'})
    test['prediction_datetime'] = pd.to_datetime(test['prediction_datetime'])
    test = pl.from_pandas(test)
    test = test.with_columns(
                (pl.col('prediction_datetime').dt.date() +
                    datetime.timedelta(days=-2)).alias('date_when_predicting')
              )

    # client formatting
    client['date'] = pd.to_datetime(client['date']).dt.date
    client = pl.from_pandas(client)

    #return (test, revealed_targets, client, 
    #        weatherHistorical, weatherForecast,
    #        electricityPrices, gasPrices, sample_prediction)

    return test, client


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
