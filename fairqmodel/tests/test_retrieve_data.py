"""Tests for retrieve_data module.

Unit tests for the pure helper functions (no DB required).
The retrieve_data() and retrieve_cap_values() functions themselves require
a live ClickHouse connection with the production schema and are therefore
only tested manually (@pytest.mark.tests_on_real_clickhouse).
"""

import pandas as pd

from fairqmodel.retrieve_data import check_number_of_rows, fill_in_date_min_and_max

# ---------------------------------------------------------------------------
# fill_in_date_min_and_max
# ---------------------------------------------------------------------------


def test_fill_in_date_min_and_max_both_provided():
    """Check that provided timestamps are converted to UTC without modification of values."""
    ts_min = pd.Timestamp("2023-01-01 00:00:00", tz="Europe/Berlin")
    ts_max = pd.Timestamp("2023-01-10 00:00:00", tz="Europe/Berlin")
    result_min, result_max = fill_in_date_min_and_max(ts_min, ts_max, include_future_data=False)
    assert result_min.tzinfo is not None
    assert result_max.tzinfo is not None
    assert str(result_min.tzinfo) == "UTC"
    assert str(result_max.tzinfo) == "UTC"


def test_fill_in_date_min_and_max_none_min_defaults_to_2015():
    """Check that None min date defaults to 2015-01-01 UTC."""
    ts_max = pd.Timestamp("2023-01-10 00:00:00", tz="Europe/Berlin")
    result_min, _ = fill_in_date_min_and_max(None, ts_max, include_future_data=False)
    assert result_min == pd.Timestamp("2015-01-01", tz="UTC")


def test_fill_in_date_min_and_max_none_max_with_future():
    """Check that None max date with future data yields a timestamp approximately 5 days ahead."""
    _, result_max = fill_in_date_min_and_max(None, None, include_future_data=True)
    now_utc = pd.Timestamp.now(tz="UTC")
    # max should be approximately now + 5 days
    assert result_max > now_utc
    assert result_max < now_utc + pd.Timedelta(7, "day")


def test_fill_in_date_min_and_max_none_max_without_future():
    """Check that None max date without future data yields approximately the current time."""
    _, result_max = fill_in_date_min_and_max(None, None, include_future_data=False)
    now_utc = pd.Timestamp.now(tz="UTC")
    # max should be approximately now (+ 0 days)
    assert result_max <= now_utc + pd.Timedelta(1, "hour")


def test_fill_in_date_min_and_max_converts_to_utc():
    """Check that Berlin timestamps are correctly converted to UTC."""
    ts_min = pd.Timestamp("2023-06-01 12:00:00", tz="Europe/Berlin")
    ts_max = pd.Timestamp("2023-06-10 12:00:00", tz="Europe/Berlin")
    result_min, result_max = fill_in_date_min_and_max(ts_min, ts_max, include_future_data=False)
    assert str(result_min.tzinfo) == "UTC"
    assert str(result_max.tzinfo) == "UTC"
    assert result_min.hour == 10  # noqa: PLR2004 - Berlin summer time is UTC+2, so 12:00 Berlin = 10:00 UTC
    assert result_max.hour == 10  # noqa: PLR2004


# ---------------------------------------------------------------------------
# check_number_of_rows
# ---------------------------------------------------------------------------


def _make_dat(n_coords: int, n_hours: int) -> pd.DataFrame:
    """Build a DataFrame with n_coords unique (x, y) pairs and n_hours rows each."""
    rows = [{"x": float(i), "y": 0.0, "value": h} for i in range(n_coords) for h in range(n_hours)]
    return pd.DataFrame(rows)


def test_check_number_of_rows_exact_match_returns_true():
    """Check that exact row count returns True."""
    ts_min = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
    ts_max = pd.Timestamp("2023-01-01 02:00:00", tz="UTC")  # 3 hours incl.
    dat = _make_dat(n_coords=2, n_hours=3)
    result = check_number_of_rows(batch=None, mode="stations", dat=dat, date_time_min=ts_min, date_time_max=ts_max)
    assert result is True


def test_check_number_of_rows_mismatch_returns_false():
    """Check that mismatched row count returns False."""
    ts_min = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
    ts_max = pd.Timestamp("2023-01-01 04:00:00", tz="UTC")  # 5 hours expected
    dat = _make_dat(n_coords=2, n_hours=3)  # only 3 hours -> mismatch
    result = check_number_of_rows(batch=None, mode="stations", dat=dat, date_time_min=ts_min, date_time_max=ts_max)
    assert result is False
