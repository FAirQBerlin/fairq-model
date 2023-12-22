from datetime import datetime

import pandas as pd
import pytz

from fairqmodel.time_handling import to_unix_datetime, to_unix_series


def test_to_unix_series():
    # Arrange
    date_winter = "2022-11-22 09:22:49"
    date_summer = "2022-06-22 09:22:49"
    dat = pd.DataFrame(
        {
            "date_column_naive": [
                pd.Timestamp(date_summer),
                pd.Timestamp(date_winter),
            ],
            "date_column_utc": [
                pd.Timestamp(date_summer, tz="UTC"),
                pd.Timestamp(date_winter, tz="UTC"),
            ],
            "date_column_berlin": [
                pd.Timestamp(date_summer, tz="Europe/Berlin"),
                pd.Timestamp(date_winter, tz="Europe/Berlin"),
            ],
        }
    )
    # Calculated with 'https://unixtime.org/'
    date_target_utc_winter = 1669108969
    date_target_berlin_winter = 1669105369

    date_target_utc_summer = 1655889769
    date_target_berlin_summer = 1655882569

    # Act
    res_naive = to_unix_series(dat["date_column_naive"])
    res_utc = to_unix_series(dat["date_column_utc"])
    res_berlin = to_unix_series(dat["date_column_berlin"])

    # Assert
    # Summer time
    assert date_target_utc_summer == res_naive.values[0]
    assert date_target_utc_summer == res_utc.values[0]
    assert date_target_berlin_summer == res_berlin.values[0]

    # Winter time
    assert date_target_utc_winter == res_naive.values[1]
    assert date_target_utc_winter == res_utc.values[1]
    assert date_target_berlin_winter == res_berlin.values[1]

    # Difference between Berlin and UTC
    # Winter time
    assert (res_naive.values[1] - res_berlin.values[1]) == 3600

    # Summer time
    assert (res_naive.values[0] - res_berlin.values[0]) == 7200


def test_to_unix_datetime():
    # Arrange
    date_naive_summer = datetime(2022, 6, 22, 9, 22, 49)
    date_naive_winter = datetime(2022, 11, 22, 9, 22, 49)

    date_utc_summer = datetime(2022, 6, 22, 9, 22, 49, tzinfo=pytz.utc)
    date_utc_winter = datetime(2022, 11, 22, 9, 22, 49, tzinfo=pytz.utc)

    # NOTE:
    # datetime(.., tzinfo = pytz.some_time_zone) doesn't work for time zones with daylight saving, e.g. "Europe/Berlin"
    # See: 'https://pytz.sourceforge.net/#localized-times-and-date-arithmetic'
    date_berlin_summer = pytz.timezone("Europe/Berlin").localize(datetime(2022, 6, 22, 9, 22, 49))
    date_berlin_winter = pytz.timezone("Europe/Berlin").localize(datetime(2022, 11, 22, 9, 22, 49))

    # Calculated with 'https://unixtime.org/'
    date_target_utc_winter = 1669108969
    date_target_berlin_winter = 1669105369

    date_target_utc_summer = 1655889769
    date_target_berlin_summer = 1655882569

    # Act
    res_naive_summer = to_unix_datetime(date_naive_summer)
    res_naive_winter = to_unix_datetime(date_naive_winter)
    res_utc_summer = to_unix_datetime(date_utc_summer)
    res_utc_winter = to_unix_datetime(date_utc_winter)
    res_berlin_summer = to_unix_datetime(date_berlin_summer)
    res_berlin_winter = to_unix_datetime(date_berlin_winter)

    # Assert
    # Summer time
    assert date_target_utc_summer == res_naive_summer
    assert date_target_utc_summer == res_utc_summer
    assert date_target_berlin_summer == res_berlin_summer

    # Winter time
    assert date_target_utc_winter == res_naive_winter
    assert date_target_utc_winter == res_utc_winter
    assert date_target_berlin_winter == res_berlin_winter

    # Difference between Berlin and UTC
    # Winter time
    assert (res_naive_winter - res_berlin_winter) == 3600

    # Summer time
    assert (res_naive_summer - res_berlin_summer) == 7200
