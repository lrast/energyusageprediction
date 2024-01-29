# makes additional data sets
import pandas as pd
from astral.location import LocationInfo, Location


def make_solar_data(daterange=['2021-09-01 00:00:00', '2024-12-31 23:00:00']):
    """ make a dataset of solar positions """
    estonia_location = Location(LocationInfo(
                                    name='Estonia',
                                    region='Estonia',
                                    latitude=58.60, longitude=25.01,
                                    timezone="Europe/Tallinn"
                                ))

    all_times = pd.date_range(daterange[0], daterange[1], freq='H').to_series()
    all_times.index = all_times.index.rename('datetime')

    pd.DataFrame({
                'solar_azimuth': all_times.apply(estonia_location.solar_azimuth),
                'solar_elevation': all_times.apply(estonia_location.solar_elevation)
                }).to_csv('enefittools/data/datasets/solar_data.csv')


def make_rich_solar_data(locations,
                         daterange=['2021-09-01 00:00:00', '2024-12-31 23:00:00']):
    """Dataset of solar data at all of the different locations"""

    all_times = pd.date_range(daterange[0], daterange[1], freq='H').to_series()
    all_times.index = all_times.index.rename('datetime')

    data_for_locations = []

    for latitude, longitude in locations:
        current_location = Location(LocationInfo(
                                    name=f'loc:{latitude},{longitude}',
                                    region='Estonia',
                                    latitude=latitude, longitude=longitude,
                                    timezone="Europe/Tallinn"
                                    ))
        elevations = all_times.apply(current_location.solar_elevation)
        azimuths = all_times.apply(current_location.solar_azimuth)

        solar_data = pd.DataFrame({'solar_elevation': elevations,
                                   'solar_azimuth': azimuths}).reset_index()
        location_data = pd.DataFrame({'latitude': [latitude],
                                      'longitude': [longitude]})
        full_data = solar_data.merge(location_data, how='cross')
        data_for_locations.append(full_data)
        print(len(data_for_locations))

    pd.concat(data_for_locations).to_csv('enefittools/data/datasets/solar_data_full.csv',
                                         index=False)
