from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from fairqmodel.build_splits import prepare_folded_input, set_prediction_hour


def test_prepare_folded_input():
    """
    Checks the returned format, size and content.
    """

    # Arrange
    time_points = np.arange(datetime(2015, 4, 1), datetime(2019, 12, 24), timedelta(hours=1)).astype(datetime)
    dat = pd.DataFrame({"date_time": time_points})
    n_windows = 90
    n_train_years = 1
    test_window_size = 24
    step_size = 12
    n_train_hours = n_train_years * 365 * 24

    target_keys = {
        "ts_fold_id",
        "ts_fold_max_train_date",
        "ts_fold_max_test_date",
        "train",
        "test",
    }

    # Act
    res = prepare_folded_input(dat, n_windows, n_train_years, test_window_size, step_size)
    res_2 = prepare_folded_input(
        dat, n_windows, n_train_years, test_window_size, step_size, include_current_time_point=True
    )

    # Assert
    assert isinstance(res, list)
    assert all(isinstance(x, dict) for x in res)
    assert all(set(x.keys()) == target_keys for x in res)
    assert all(x["train"].shape[0] <= n_train_hours for x in res)  # smaller equal to account for missing data
    assert all(x["test"].shape[0] <= test_window_size for x in res)  # smaller equal to account for missing data
    assert all(x["test"].shape[0] <= test_window_size + 1 for x in res_2)  # smaller equal to account for missing data
    assert all(
        x["ts_fold_max_test_date"] - x["ts_fold_max_train_date"] == timedelta(hours=test_window_size) for x in res
    )


def test_set_prediction_hour():
    # Arrange
    max_date_winter = datetime(2022, 11, 11, 11, 11, 11)  # in UTC but tz naive
    max_date_summer = datetime(2022, 6, 11, 11, 11, 11)  # in UTC but tz naive

    prediction_window_size = 46

    # Prediction hours in Berlin Time
    prediction_hour_later = 17
    prediction_hour_earlier = 9
    prediction_hour_correct_winter = 14
    prediction_hour_correct_summer = 15

    # What are the correct 'max_dates' (UTC) for predictions of 'prediction_window_size' hours
    # starting at 'prediction_hour'(Berlin Time):
    # Winter
    max_date_target_later_winter = datetime(2022, 11, 10, 14, 11, 11)  # = '2022-11-08 16:11:11' + 46h
    max_date_target_earlier_winter = datetime(2022, 11, 11, 6, 11, 11)  # = '2022-11-09 08:11:11' + 46h
    max_date_target_correct_winter = datetime(2022, 11, 11, 11, 11, 11)  # = '2022-11-09 13:11:11' + 46h

    # Summer
    max_date_target_later_summer = datetime(2022, 6, 10, 13, 11, 11)  # = '2022-06-08 15:11:11' + 46h
    max_date_target_earlier_summer = datetime(2022, 6, 11, 5, 11, 11)  # = '2022-06-09 07:11:11' + 46h
    max_date_target_correct_summer = datetime(2022, 6, 11, 11, 11, 11)  # = '2022-06-09 13:11:11' + 46h

    # Act
    # Winter
    res_later_winter = set_prediction_hour(prediction_hour_later, max_date_winter, prediction_window_size)
    res_earlier_winter = set_prediction_hour(prediction_hour_earlier, max_date_winter, prediction_window_size)
    res_correct_winter = set_prediction_hour(prediction_hour_correct_winter, max_date_winter, prediction_window_size)

    # Summer
    res_later_summer = set_prediction_hour(prediction_hour_later, max_date_summer, prediction_window_size)
    res_earlier_summer = set_prediction_hour(prediction_hour_earlier, max_date_summer, prediction_window_size)
    res_correct_summer = set_prediction_hour(prediction_hour_correct_summer, max_date_summer, prediction_window_size)

    # Assert
    # Correct date winter
    assert max_date_target_later_winter == res_later_winter
    assert max_date_target_earlier_winter == res_earlier_winter
    assert max_date_target_correct_winter == res_correct_winter

    # Correct date summer
    assert max_date_target_later_summer == res_later_summer
    assert max_date_target_earlier_summer == res_earlier_summer
    assert max_date_target_correct_summer == res_correct_summer

    # Correct format winter
    assert res_later_winter.tzinfo is None
    assert res_earlier_winter.tzinfo is None
    assert res_correct_winter.tzinfo is None

    # Correct format summer
    assert res_later_summer.tzinfo is None
    assert res_earlier_summer.tzinfo is None
    assert res_correct_summer.tzinfo is None
