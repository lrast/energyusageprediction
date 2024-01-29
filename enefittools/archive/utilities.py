import pandas as pd
import datetime


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
