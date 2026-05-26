"""Unit tests for get_max_date_time module."""

import pandas as pd

from fairqmodel.get_max_date_time import get_max_date_time


def test_returns_timestamp():
    """Check that get_max_date_time returns a pd.Timestamp."""
    ts = pd.Timestamp("2023-06-01 10:00:00", tz="Europe/Berlin")
    result = get_max_date_time(ts)
    assert isinstance(result, pd.Timestamp)


def test_result_is_berlin_timezone():
    """Check that the result is in Europe/Berlin timezone."""
    ts = pd.Timestamp("2023-06-01 10:00:00", tz="Europe/Berlin")
    result = get_max_date_time(ts)
    assert str(result.tzinfo) == "Europe/Berlin"


def test_result_hour_is_midnight():
    """Check that the result timestamp has hour set to 0 (midnight)."""
    ts = pd.Timestamp("2023-06-01 10:00:00", tz="Europe/Berlin")
    result = get_max_date_time(ts)
    assert result.hour == 0


def test_before_noon_returns_4_days_ahead():
    """Check that input before noon returns max_date 4 days ahead."""
    # hour < 12 -> floor_days = 4
    ts = pd.Timestamp("2023-06-01 10:00:00", tz="Europe/Berlin")
    result = get_max_date_time(ts)
    expected_date = (ts + pd.Timedelta(4, "day")).replace(hour=0)
    assert result.date() == expected_date.date()


def test_after_noon_returns_5_days_ahead():
    """Check that input after noon returns max_date 5 days ahead."""
    # hour >= 12 -> floor_days = 5
    ts = pd.Timestamp("2023-06-01 13:00:00", tz="Europe/Berlin")
    result = get_max_date_time(ts)
    expected_date = (ts + pd.Timedelta(5, "day")).replace(hour=0)
    assert result.date() == expected_date.date()


def test_none_input_uses_current_time():
    """Check that None input falls back to the current time."""
    result = get_max_date_time(None)
    assert isinstance(result, pd.Timestamp)
    assert result.tzinfo is not None
