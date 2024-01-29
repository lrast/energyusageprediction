import datetime
import polars as pl
import pandas as pd


def format_dfs(target=None, revealed_targets=None, client=None,
               weather_historical=None, weather_forecast=None,
               electricity_prices=None, gas_prices=None, 
               sample_prediction=None, solar=None,

               filter_weather=True,
               assemble_and_split=False,
               mode='train'):
    """ dataframe formatting for training and online use 
        
        - encodes as polars dataframes
        - sets proper types
        - filter out two weather stations that don't make sense (optional)
        - join target and client data (optional)
        - split by production, consumption (optional)
    """

    column_types = {'county': pl.Int8, 'product_type': pl.Int8,
                    'is_business': pl.Boolean, 'is_consumption': pl.Boolean,
                    'row_id': pl.Int64, 'prediction_unit_id': pl.Int8,
                    'data_block_id': pl.Int64,

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

    if client is not None:
        # client formatting
        client['date'] = pd.to_datetime(client['date']).dt.date
        client = pl.from_pandas(client, schema_overrides=column_types)
        client = client.with_columns()

    if weather_historical is not None:
        weather_historical['datetime'] = pd.to_datetime(weather_historical['datetime'])
        weather_historical = pl.from_pandas(weather_historical,
                                            schema_overrides=column_types)
        if filter_weather:
            weather_historical = weather_historical.filter(
                                    ((pl.col('latitude') != 57.6) | (pl.col('longitude') != 23.2)),
                                    ((pl.col('latitude') != 57.6) | (pl.col('longitude') != 24.2)),
                                   )

    if weather_forecast is not None:
        weather_forecast['origin_datetime'] = pd.to_datetime(weather_forecast['origin_datetime'])
        weather_forecast['forecast_datetime'] = pd.to_datetime(weather_forecast['forecast_datetime'])
        weather_forecast = pl.from_pandas(weather_forecast,
                                          schema_overrides=column_types)

        if filter_weather:
            weather_forecast = weather_forecast.filter(
                                    ((pl.col('latitude') != 57.6) | (pl.col('longitude') != 23.2)),
                                    ((pl.col('latitude') != 57.6) | (pl.col('longitude') != 24.2)),
                                   )

    if electricity_prices is not None:
        electricity_prices['forecast_datetime'] = pd.to_datetime(electricity_prices['forecast_date'])
        electricity_prices['origin_datetime'] = pd.to_datetime(electricity_prices['origin_date'])
        electricity_prices = pl.from_pandas(electricity_prices, schema_overrides=column_types)
        electricity_prices = electricity_prices.drop('forecast_date', 'origin_date'
                                              ).with_columns(
                                                    orgin_date=pl.col('origin_datetime').dt.date()
                                              )

    if gas_prices is not None:
        gas_prices['forecast_date'] = pd.to_datetime(gas_prices['forecast_date']).dt.date
        gas_prices['origin_date'] = pd.to_datetime(gas_prices['origin_date']).dt.date
        gas_prices = pl.from_pandas(gas_prices, schema_overrides=column_types)

    if solar is not None:
        solar['datetime'] = pd.to_datetime(solar['datetime'])
        solar = pl.from_pandas(solar, schema_overrides=column_types)
        if 'latitude' in solar and 'longitude' in solar:
            solar = solar.filter(
                        ((pl.col('latitude') != 57.6) | (pl.col('longitude') != 23.2)),
                        ((pl.col('latitude') != 57.6) | (pl.col('longitude') != 24.2)),
                       )

    if sample_prediction is not None:
        pass

    if revealed_targets is not None:
        revealed_targets['datetime'] = pd.to_datetime(revealed_targets['datetime'])
        revealed_targets = pl.from_pandas(revealed_targets,
                                          schema_overrides=column_types)
        revealed_targets = revealed_targets.rename({'datetime': 'prediction_datetime'})
        revealed_targets = assemble_train_client(revealed_targets, client, 'train')

    if assemble_and_split:
        prod, consume = split_prod_consume(assemble_train_client(target, client, mode))

        return (prod, consume) + \
            tuple(filter(lambda x: x is not None,
                         (revealed_targets, weather_historical, weather_forecast,
                          electricity_prices, gas_prices, sample_prediction, solar)
                         ))

    else:
        return tuple(filter(lambda x: x is not None,
                            (target, revealed_targets, client, 
                             weather_historical, weather_forecast,
                             electricity_prices, gas_prices, sample_prediction,
                             solar
                             )))


def assemble_train_client(train, client, mode):
    """Join the train and client datasets together into one dataset."""

    # I want to train with the ground truth installed capacity, even if this
    # is not available at test time
    if mode == 'train':
        return train.with_columns(
                        pl.col('prediction_datetime').dt.date().alias('date')
                    ).join(client,
                           on=['county', 'product_type', 'is_business', 'date']
                    )
    else:
        # when actively predicting, we only have client data from two days
        # previous
        return train.with_columns(
                        pl.col('prediction_datetime').dt.date().alias('date')
                    ).join(client,
                           left_on=['county', 'product_type', 'is_business', 'date_when_predicting'],
                           right_on=['county', 'product_type', 'is_business', 'date']
                    )


def split_prod_consume(targets):
    return (targets.filter(pl.col('is_consumption') == False),
            targets.filter(pl.col('is_consumption') == True))
