from datetime import datetime

import pandas as pd
from pandas.testing import assert_frame_equal
from pytest import raises

from fairqmodel.data_preprocessing import (
    cap_high_values,
    cap_outliers,
    drop_stations_without_this_depvar,
    fix_column_types,
)


def test_fix_column_types_columns():
    """
    Checks if feature names are in DataFrame.
    """
    # arrange
    dat = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4], "date_time": [1, 2]})
    categorical_feature_cols = ["col1"]
    metric_feature_cols = ["col42"]

    # act/assert
    with raises(KeyError):
        fix_column_types(dat, categorical_feature_cols, metric_feature_cols)


def test_fix_column_type_output():
    """
    Checks the returend DataFrame for data and correct dtype.
    """
    # arrange
    date_now = datetime.now()
    dat = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4], "date_time": date_now})
    expected = pd.DataFrame(
        data={"col1": pd.Series([1, 2]).astype("category"), "col2": [3.0, 4.0], "date_time": pd.to_datetime(date_now)}
    )
    categorical_feature_cols = ["col1"]
    metric_feature_cols = ["col2"]

    # act
    res = fix_column_types(dat, categorical_feature_cols, metric_feature_cols)

    # assert
    pd.testing.assert_frame_equal(res, expected)


def test_drop_stations_without_this_depvar():
    input_df = pd.DataFrame(
        {"station_id": [1, 1, 2, 2, 3], "depvar": [None, None, 5.0, 6, 7], "not_the_depvar": [1.0, 2, None, None, 4]}
    )
    expected = pd.DataFrame({"station_id": [2, 2, 3], "depvar": [5.0, 6, 7], "not_the_depvar": [None, None, 4]})
    res = drop_stations_without_this_depvar(input_df, "depvar").reset_index(drop=True)
    assert_frame_equal(res, expected)


def test_drop_stations_without_this_depvar_no_change():
    input_df = pd.DataFrame({"station_id": [1, 1, 2, 2, 3], "depvar": range(5)})
    res = drop_stations_without_this_depvar(input_df, "depvar").reset_index(drop=True)
    assert_frame_equal(res, input_df)


def test_remove_outliers():
    # Arrange
    dummy_data = [1, 2, 3, 4, 5]
    dat = pd.DataFrame({"no2": dummy_data, "pm10": dummy_data, "pm25": dummy_data})

    cap_values = pd.DataFrame(
        {
            "pollutant": ["no2", "pm10", "pm25"],
            "cap_value_outlier": [2, 3, 4],
            "cap_value_training": [1, 2, 3],
        }
    )

    target_no2 = [1, 2, 2, 2, 2]
    target_npm10 = [1, 2, 3, 3, 3]
    target_pm25 = [1, 2, 3, 4, 4]

    # Act
    dat_res = cap_outliers(dat, cap_values)

    # Assert
    assert all(target_no2 == dat_res.no2)
    assert all(target_npm10 == dat_res.pm10)
    assert all(target_pm25 == dat_res.pm25)


def test_cap_high_values():
    # Arrange
    dummy_data = [1, 2, 3, 4, 5, 6]
    dat = pd.DataFrame(
        {
            "no2": dummy_data,
            "no2_lag_981": dummy_data,
            "no2_sth": dummy_data,
            "pm25": dummy_data,
            "lag_avg(1,2,3)": dummy_data,
            "cams_no2_la4": dummy_data,
        }
    )
    depvar = "no2"

    cap_values = pd.DataFrame(
        {
            "pollutant": ["no2", "pm10", "pm25"],
            "cap_value_outlier": [2, 3, 4],
            "cap_value_training": [3, 3, 3],
        }
    )
    target_data_capped = [1, 2, 3, 3, 3, 3]
    target_data_uncapped = dummy_data
    dat_target = pd.DataFrame(
        {
            "no2": target_data_capped,
            "no2_lag_981": target_data_capped,
            "no2_sth": target_data_uncapped,
            "pm25": target_data_uncapped,
            "lag_avg(1,2,3)": target_data_capped,
            "cams_no2_la4": target_data_uncapped,
        }
    )

    # Act
    dat_res = cap_high_values(dat, depvar, cap_values)

    # Assert
    assert_frame_equal(dat_target, dat_res)
