"""Unit tests for time_handling module."""

import pandas as pd

from fairqmodel.time_handling import (
    get_current_local_time,
    get_model_start_time,
    timestamp_to_tz_aware,
)


def test_get_current_local_time_returns_berlin_time():
    """Check that get_current_local_time returns a timezone-aware Berlin datetime."""
    result = get_current_local_time()
    assert result.tzinfo is not None
    assert result.tzinfo.zone == "Europe/Berlin"


def test_get_current_local_time_no_microseconds():
    """Check that get_current_local_time returns a datetime with no microseconds."""
    result = get_current_local_time()
    assert result.microsecond == 0


def test_get_model_start_time_returns_timestamp():
    """Check that get_model_start_time returns a pd.Timestamp."""
    result = get_model_start_time()
    assert isinstance(result, pd.Timestamp)


def test_get_model_start_time_twice_daily_is_5_or_15():
    """Check that twice_daily model start time is either 5 or 15."""
    result = get_model_start_time(twice_daily=True)
    assert result.hour in (5, 15)


def test_get_model_start_time_not_twice_daily_returns_current_hour():
    """Check that non-twice-daily model start time matches the current hour."""
    result = get_model_start_time(twice_daily=False)
    current_hour = get_current_local_time().replace(second=0, minute=0).hour
    assert result.hour == current_hour


def test_timestamp_to_tz_aware_naive_utc():
    """Check that a tz-naive timestamp is localized to UTC and converted to Berlin time."""
    ts = pd.Timestamp("2023-01-01 12:00:00")
    result = timestamp_to_tz_aware(ts)
    assert result.tzinfo is not None
    assert str(result.tzinfo) == "Europe/Berlin"


def test_timestamp_to_tz_aware_already_tz_aware():
    """Check that a UTC-aware timestamp is converted to Berlin time."""
    ts = pd.Timestamp("2023-01-01 12:00:00", tz="UTC")
    result = timestamp_to_tz_aware(ts)
    assert result.tzinfo is not None
    assert str(result.tzinfo) == "Europe/Berlin"


def test_timestamp_to_tz_aware_utc_offset_winter():
    """Check that UTC+0 is correctly converted to UTC+1 in Berlin winter time."""
    # In winter Berlin is UTC+1
    ts = pd.Timestamp("2023-01-01 10:00:00", tz="UTC")
    result = timestamp_to_tz_aware(ts)
    assert result.hour == 11  # noqa: PLR2004 - Berlin winter time is UTC+1, so 10:00 UTC = 11:00 Berlin
