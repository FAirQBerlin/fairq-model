from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
import pytz


def to_unix(entry: Union[pd.Series, datetime]) -> Union[pd.Series, int]:
    """Converts the provided entry into unix format.
    The entry can either be a Series of dates or a datetime object.

    :param entry: Union[pd.Series, datetime],
                  pd.Series: A column of a DataFrame containing dates
                  datetime: A single date in datetime format

    :return: Union[pd.Series, int],
                  pd.Series: Same series but in unix format
                  int: Unix timestamp of the datetime
    """
    assert isinstance(
        entry, (pd.Series, datetime)
    ), "Provided entry has incorrect type and can't be converted to unix format"

    if isinstance(entry, pd.Series):
        result = to_unix_series(entry)
    else:
        result = to_unix_datetime(entry)

    return result


def to_unix_series(col: pd.Series) -> pd.Series:
    """Converts a Series of dates to unix format

    :param col:pd.Series, Column of a DataFrame containing dates

    :return:pd.Series, Same column but in unix format
    """
    # If the series is tz naive, it is interpreted as utc
    if col.dt.tz is None:
        col = col.dt.tz_localize("utc")

    col_unix = col.apply(to_unix_datetime)

    return col_unix


def to_unix_datetime(date: datetime) -> int:
    """Converts a datetime object to unix format

    :param date:datetime, Date to be converted

    :return:int, Date in unix format
    """
    # If the date is tz naive, it is interpreted as utc
    if date.tzinfo is None:
        date = pytz.utc.localize(date)

    date_unix = int(date.timestamp())

    return date_unix


def get_current_local_time() -> datetime:
    """Returns the current date in Berlin time"""

    utc_now = datetime.utcnow().replace(tzinfo=pytz.utc)
    berlin_now = utc_now.astimezone(pytz.timezone("Europe/Berlin")).replace(microsecond=0)

    return berlin_now


def get_model_start_time(twice_daily: bool = True) -> pd.Timestamp:
    """Gets the model start time.
    For models that run twice a day, the start time is either 5 a.m. or 3 p.m.
    For models that run hourly, the date with the current hour is returned.

    :twice_daily: bool, Specifies if the date is selected for a twice-a-day or an hourly model

    :return: pd.Timestamp
    """

    berlin_now = get_current_local_time().replace(second=0, minute=0)

    if twice_daily:
        if berlin_now.hour >= 15:
            date_time_forecast = berlin_now.replace(hour=15)
        elif berlin_now.hour < 5:
            previous_day = berlin_now - pd.Timedelta(1, "day")
            date_time_forecast = previous_day.replace(hour=15)
        else:  # berlin_now.hour < 15 or berlin_now.hour >= 5
            date_time_forecast = berlin_now.replace(hour=5)
    else:  # not twice_daily -> return date with current hour
        date_time_forecast = berlin_now

    return pd.Timestamp(date_time_forecast)


def timestamp_to_np_datetime64(date: pd.Timestamp) -> np.datetime64:
    """Casts a Timestamp to np.datetime64 format. Before casting, the Timestamp is converted to UTC.
    If it is already in UTC the conversion has no effect.
    If the Timestamp has no tzinfo, it is assumed to be in UTC.

    :param date: pd.Timestamp, Date in an arbitrary time zone

    :return: np.datetime64, Date in UTC as np.datetime64 format
    """

    if date.tzinfo is not None:
        date = date.tz_convert(tz="UTC").tz_localize(None)

    date_as_np_64 = np.datetime64(date)

    return date_as_np_64
