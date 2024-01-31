# adds weather data to the features

from sklearn.base import BaseEstimator, TransformerMixin


class Simple_Weather_Features(BaseEstimator, TransformerMixin):
    """Transformer that adds Solar Features to the data

        This is a simple version that uses only weather stations
        in the corresponding county
    """
    def __init__(self, county_mapping):
        super(Simple_Weather_Features, self).__init__()
        self.county_mapping = county_mapping

    def fit(self, data_holder, y=None):
        self.is_fit_ = True
        return self

    def transform(self, data_holder, y=None):
        relevant_columns = ['temperature', 'dewpoint', 'cloudcover_high', 'cloudcover_low',
                            'cloudcover_mid', 'cloudcover_total', '10_metre_u_wind_component',
                            '10_metre_v_wind_component', 'forecast_datetime', 'county',
                            'direct_solar_radiation', 'surface_solar_radiation_downwards',
                            'snowfall', 'total_precipitation']

        weather_selection = self.county_mapping.join(
                                                data_holder.weather_forecast,
                                                on=['latitude', 'longitude'],
                                                how='left'
                                              ).fill_null(0
                                              ).select(
                                                relevant_columns
                                             )

        outs = data_holder.features.join(weather_selection,
                                         left_on=['prediction_datetime', 'county'],
                                         right_on=['forecast_datetime', 'county'])

        data_holder.features = outs

        return data_holder
