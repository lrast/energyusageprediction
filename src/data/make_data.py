# data cleaning and preprocessing
import shutil
import pandas as pd
import datetime


def clean_data():
    """missing data imputation and duplicate removal"""
    print('client')
    shutil.copy('data/raw/client.csv', 'data/processed/client.csv')

    # convert to UTC and impute missing data for electricity prices
    print('electricity_prices')
    electricityPrices = pd.read_csv('data/raw/electricity_prices.csv')
    electricityPrices = localToUTC(electricityPrices, 'forecast_date', 'Europe/Warsaw')
    electricityPrices.origin_date = electricityPrices.forecast_date - \
                                        datetime.timedelta(days=1)
    electricityPrices.sort_values('forecast_date')

    electricityPrices.to_csv('data/processed/electricity_prices.csv', index=False)

    # impute missing data for weather forcast
    # it happens that these are the same date and location
    print('forecast_weather')
    weatherForecast = pd.read_csv('data/raw/forecast_weather.csv')
    subSelection = weatherForecast[(weatherForecast.latitude == 59.7) & 
                (weatherForecast.longitude == 23.7) & 
                (weatherForecast.origin_datetime == '2022-08-11 00:00:00+00:00')]
    fillVal = (
        subSelection[subSelection.hours_ahead == 2].surface_solar_radiation_downwards
        +
        subSelection[subSelection.hours_ahead == 5].surface_solar_radiation_downwards
        )/2
    weatherForecast.fillna(fillVal)
    weatherForecast.to_csv('data/processed/forecast_weather.csv', index=False)

    print('gas_prices')
    shutil.copy('data/raw/gas_prices.csv', 'data/processed/gas_prices.csv')

    # remove duplicates from historical weather
    print('historical_weather')
    weatherHistorical = pd.read_csv('data/raw/historical_weather.csv')
    weatherHistorical.drop_duplicates()
    weatherHistorical.datetime = pd.to_datetime(weatherHistorical.datetime
                                                ).dt.tz_localize('UTC')
    weatherHistorical.to_csv('data/processed/historical_weather.csv', index=False)

    print('train')
    trainData = pd.read_csv('data/raw/train.csv')
    # note the strange behavior of gmt -2, which converts to utc+2...
    trainData.datetime = pd.to_datetime(trainData.datetime). \
                            dt.tz_localize('Etc/GMT-2').dt.tz_convert('UTC')
    trainData.to_csv('data/processed/train.csv', index=False)


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
