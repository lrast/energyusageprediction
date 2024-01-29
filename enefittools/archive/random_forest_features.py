import polars as pl


def make_features_random_forest(target, client, weather_forecast, solar):
    """ for the moment, we'll just use the forecast features """

    # note that this gives us two sets of features for each time: 
    # it is a daily 48hr forecast
    weather_features = weather_forecast.with_columns(
                                    pl.concat_str(pl.col('latitude'), pl.col('longitude'), separator=',').alias('location')
                               ).pivot(
                                        index=['forecast_datetime', 'hours_ahead'],
                                        columns=['location'],
                                        values=['temperature', 'direct_solar_radiation']
                               ).drop('hours_ahead')
    unit_features = target.drop('data_block_id'
                         ).join(
                              client.drop('data_block_id'), 
                              left_on=['county', 'is_business', 'product_type', 'date_when_predicting'],
                              right_on=['county', 'is_business', 'product_type', 'date'],
                                how='inner'
                         )
    return unit_features.join(weather_features, 
                              left_on='prediction_datetime', 
                              right_on='forecast_datetime', how='inner'
                       ).join(
                            solar, left_on='prediction_datetime', 
                            right_on='datetime', how='inner'
                       ).drop(['target', 'prediction_datetime', 'date_when_predicting'])