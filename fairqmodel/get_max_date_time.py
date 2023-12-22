from typing import Optional

import pandas as pd
import tzlocal

from fairqmodel.time_handling import get_current_local_time


def get_max_date_time(
    date_now: Optional[pd.Timestamp] = None,
) -> pd.Timestamp:
    """Gets the maximal 'date_time' for a prediction

    :date_now: pd.Timestamp, Current time stamp floored to the hour

    :return: pd.Timestamp, In Berlin Time
    """
    if date_now is None:
        date_now = pd.Timestamp(get_current_local_time().replace(second=0, minute=0))

    assert date_now is not None

    # Ensure that the timestamp is in local time
    if date_now.tzinfo is None:
        date_now = date_now.tz_localize(tz=tzlocal.get_localzone_name())

    # NOTE: tz_convert is idempotent,
    # i.e. If date_now is already in Berlin Time, the conversion has no effect
    date_now_berlin = date_now.tz_convert(tz="Europe/Berlin")

    floor_days = 5 if date_now.hour >= 12 else 4

    # Select max_date s.t. it is the end of the third (or fourth) day from now (0 o'clock the next day)
    max_date_time_berlin = (date_now_berlin + pd.Timedelta(floor_days, "day")).replace(hour=0)

    return max_date_time_berlin
