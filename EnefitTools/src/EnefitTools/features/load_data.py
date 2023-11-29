import pandas as pd


class DataHolder(object):
    """DataHolder: a root object that data sets are attached to """
    def __init__(self, **kwargs):
        super(DataHolder, self).__init__()
        self.__dict__.update(**kwargs)


def load_raw_data(dataPath='data'):
    """ load the raw data from csv"""

    electricityPrices = pd.read_csv(f'{dataPath}/raw/electricity_prices.csv')
    gasPrices = pd.read_csv(f'{dataPath}/raw/gas_prices.csv')
    clients = pd.read_csv(f'{dataPath}/raw/client.csv')
    train = pd.read_csv(f'{dataPath}/raw/train.csv')
    weatherHistorical = pd.read_csv(f'{dataPath}/raw/historical_weather.csv')
    weatherForecast = pd.read_csv(f'{dataPath}/raw/forecast_weather.csv')

    return DataHolder(
        electricityPrices=electricityPrices,
        gasPrices=gasPrices,
        clients=clients,
        train=train,
        weatherHistorical=weatherHistorical,
        weatherForecast=weatherForecast
    )


def load_processed_data(dataPath='data'):
    """ load the preprocessed data from csvs
    """

    electricityPrices = pd.read_csv(f'{dataPath}/processed/electricity_prices.csv',
                                    parse_dates=['forecast_date', 'origin_date'])
    gasPrices = pd.read_csv(f'{dataPath}/processed/gas_prices.csv',
                            parse_dates=['forecast_date', 'origin_date'])
    clients = pd.read_csv(f'{dataPath}/processed/client.csv',
                          parse_dates=['date'])
    train = pd.read_csv(f'{dataPath}/processed/train.csv',
                        parse_dates=['datetime'])
    weatherHistorical = pd.read_csv(f'{dataPath}/processed/historical_weather.csv',
                                    parse_dates=['datetime'])
    weatherForecast = pd.read_csv(f'{dataPath}/processed/forecast_weather.csv',
                                  parse_dates=['origin_datetime', 'forecast_datetime'])

    # For data points that are associated with days, convert timestamps to date
    gasPrices.forecast_date = gasPrices.forecast_date.dt.date
    gasPrices.origin_date = gasPrices.origin_date.dt.date

    clients.date = clients.date.dt.date

    return DataHolder(
        electricityPrices=electricityPrices,
        gasPrices=gasPrices,
        clients=clients,
        train=train,
        weatherHistorical=weatherHistorical,
        weatherForecast=weatherForecast
    )
