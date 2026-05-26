"""Unit tests for time_features module."""

import numpy as np
import pandas as pd

from fairqmodel.time_features import get_station_lags, time_features, time_lags


def make_station_df(n_rows: int = 10, depvar: str = "no2") -> pd.DataFrame:
    """Create a minimal DataFrame with one station."""
    return pd.DataFrame(
        {
            "station_id": ["s1"] * n_rows,
            "date_time": pd.date_range("2023-01-01", periods=n_rows, freq="h", tz="UTC"),
            depvar: np.arange(n_rows, dtype=float),
        }
    )


def test_time_features_no_lags_returns_unchanged_shape():
    """Check that no lags returns a DataFrame with unchanged shape."""
    dat = make_station_df()
    result = time_features(dat, depvar="no2", lags_actual=[], lags_avg=[])
    assert result.shape == dat.shape


def test_time_features_adds_lag_columns():
    """Check that lag columns are added for each specified lag."""
    dat = make_station_df()
    result = time_features(dat, depvar="no2", lags_actual=[1, 2], lags_avg=[])
    assert "no2_lag1" in result.columns
    assert "no2_lag2" in result.columns


def test_time_features_lag_values_are_shifted():
    """Check that lag values are correctly shifted by the specified number of hours."""
    dat = make_station_df()
    result = time_features(dat, depvar="no2", lags_actual=[1], lags_avg=[])
    # Row at index i should have lag1 == original value at i-1
    assert result["no2_lag1"].iloc[1] == dat["no2"].iloc[0]


def test_time_features_adds_avg_lag_column():
    """Check that an average lag column is added when lags_avg is specified."""
    dat = make_station_df()
    result = time_features(dat, depvar="no2", lags_actual=[], lags_avg=[1, 2])
    assert "lag_avg_[1, 2]" in result.columns or "lag_avg_(1, 2)" in result.columns


def test_time_lags_multiple_stations():
    """Check that lag values do not bleed across different stations."""
    dat = pd.DataFrame(
        {
            "station_id": ["s1"] * 5 + ["s2"] * 5,
            "date_time": pd.date_range("2023-01-01", periods=5, freq="h", tz="UTC").tolist() * 2,
            "no2": np.arange(10, dtype=float),
        }
    )
    result = time_lags(dat, depvar="no2", lags_actual=[1])
    # Lag values should not bleed across stations
    s2_lag1_first = result[result["station_id"] == "s2"]["no2_lag1"].iloc[0]
    assert pd.isna(s2_lag1_first)


def test_get_station_lags_first_row_is_nan():
    """Check that the first row of a lag column is NaN since there is no previous value."""
    dat = make_station_df().drop(columns=["station_id"])
    result = get_station_lags(dat, depvar="no2", lags=[1])
    assert pd.isna(result["no2_lag1"].iloc[0])
