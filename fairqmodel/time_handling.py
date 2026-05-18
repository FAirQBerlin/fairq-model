from datetime import UTC, datetime

import pandas as pd
import pytz


def get_current_local_time() -> datetime:
    """Returns the current date in Berlin time"""

    utc_now = datetime.now(UTC).replace(tzinfo=pytz.utc)
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


def timestamp_to_tz_aware(date: pd.Timestamp) -> pd.Timestamp:
    """Turns a timestamp into a timezone-aware timestamp in Europe/Berlin timezone.

    :param date: pd.Timestamp, Date in an arbitrary time zone or tz naive

    :return: pd.Timestamp, Date in Europe/Berlin timezone
    """
    if date.tzinfo is None:
        return date.tz_localize("UTC").tz_convert("Europe/Berlin")
    return date.tz_convert("Europe/Berlin")
