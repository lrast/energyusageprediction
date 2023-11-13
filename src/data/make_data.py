# data cleaning and preprocessing
import shutil
import pandas as pd
import datetime


def clean_data():
    """missing data imputation and duplicate removal"""
    shutil.copy('data/raw/client.csv', 'data/processed/client.csv')
    print('client')

    # impute missing data for electricity prices
    electricityPrices = pd.read_csv('data/raw/electricity_prices.csv')
    forecastTimes = pd.to_datetime(electricityPrices.forecast_date).sort_values()
    fullDateRange = pd.date_range(forecastTimes.min(), forecastTimes.max(),
                                  freq='H')
    missingForecastTimes = fullDateRange.difference(forecastTimes)

    for missing in missingForecastTimes:
        prevInd = (missing + datetime.timedelta(hours=-1)
                   ).strftime("%Y-%m-%d %H:%M:%S")
        nextInd = (missing + datetime.timedelta(hours=1)
                   ).strftime("%Y-%m-%d %H:%M:%S")

        ForecastDate = missing.strftime("%Y-%m-%d %H:%M:%S")
        OriginDate = (missing + datetime.timedelta(days=-1)
                      ).strftime("%Y-%m-%d %H:%M:%S")
        BlockId = electricityPrices.data_block_id.max() + 1
        Price = (electricityPrices[electricityPrices.forecast_date == prevInd]
                                  .euros_per_mwh.item() +
                 electricityPrices[electricityPrices.forecast_date == nextInd]
                                  .euros_per_mwh.item()
                 )/2

        electricityPrices.loc[len(electricityPrices)] = {
                                'forecast_date': ForecastDate, 
                                'euros_per_mwh': Price, 
                                'origin_date': OriginDate, 
                                'data_block_id': BlockId
                                }

    electricityPrices = electricityPrices.sort_values('forecast_date')

    electricityPrices.to_csv('data/processed/electricity_prices.csv', index=False)
    print('electricity_prices')

    # impute missing data for weather forcast
    # it happens that these are the same date and location
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
    print('forecast_weather')

    shutil.copy('data/raw/gas_prices.csv', 'data/processed/gas_prices.csv')
    print('gas_prices')

    # remove duplicates from historical weather
    weatherHistorical = pd.read_csv('data/raw/historical_weather.csv')
    weatherHistorical.drop_duplicates()
    weatherHistorical.to_csv('data/processed/historical_weather.csv', index=False)
    print('historical_weather')

    shutil.copy('data/raw/train.csv', 'data/processed/train.csv')
    print('train')
