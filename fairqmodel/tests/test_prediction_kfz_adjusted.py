import pandas as pd
from pytest import fixture

from fairqmodel.prediction_kfz_adjusted import adjust_kfz_per_hour_grid, mark_hours_to_modify, pre_fill_lags
from fairqmodel.tests.helpers import arrange_for_lag_pred_tests


@fixture
def input_df_times():
    example_times = [
        "2022-01-01 03:00:00",  # 4 am in Berlin (winter time) -> traffic may not be reduced
        "2022-01-01 04:00:00",  # 5 am in Berlin (winter time) -> traffic may be reduced
        "2022-01-01 20:00:00",  # 9 pm in Berlin (winter time) -> traffic may be reduced
        "2022-01-01 21:00:00",  # 10 pm in Berlin (winter time) -> traffic may not be reduced
        "2022-07-01 02:00:00",  # 4 am in Berlin (summer time) -> traffic may not be reduced
        "2022-07-01 03:00:00",  # 5 am in Berlin (summer time) -> traffic may be reduced
        "2022-07-01 19:00:00",  # 9 pm in Berlin (summer time) -> traffic may be reduced
        "2022-07-01 20:00:00",  # 10 pm in Berlin (summer time) -> traffic may not be reduced
        # Time change:
        # They are all in the middle of the night -> traffic may not be reduced;
        # but we need to check if the time conversion works correctly and does not throw any error
        "2022-03-27 00:00:00",  # 1 am in Berlin (winter time)
        "2022-03-27 01:00:00",  # already 3 am (summer time) (2 am in Berlin in winter time)
        "2022-03-27 02:00:00",  # 4 am (summer time)
        "2022-10-30 00:00:00",  # 2 am in Berlin (summer time)
        "2022-10-30 01:00:00",  # still 2 am in Berlin (winter time)
        "2022-10-30 02:00:00",  # 3 am in berlin (winter time)
    ]
    return pd.DataFrame({"date_time": [pd.to_datetime(x) for x in example_times]})


def test_mark_hours_to_modify(input_df_times):
    expected = [False, True, True, False] * 2 + [False] * 6
    res = mark_hours_to_modify(input_df_times.date_time)
    assert res.tolist() == expected


def test_adjust_kfz_per_hour_grid():
    # Arrange
    df = pd.DataFrame(
        {
            "date_time": {
                0: pd.Timestamp("2023-06-20 13:00:00"),
                1: pd.Timestamp("2023-06-20 14:00:00"),
                2: pd.Timestamp("2023-06-20 15:00:00"),
                3: pd.Timestamp("2023-06-20 01:00:00"),
            },
            "kfz_per_hour": {0: 100.0, 1: 80.0, 2: 0.0, 3: 100.0},
        }
    )
    expected = pd.DataFrame(
        {
            "date_time": {
                0: pd.Timestamp("2023-06-20 13:00:00"),
                1: pd.Timestamp("2023-06-20 14:00:00"),
                2: pd.Timestamp("2023-06-20 15:00:00"),
                3: pd.Timestamp("2023-06-20 01:00:00"),
            },
            "kfz_per_hour": {0: 80.0, 1: 64.0, 2: 0.0, 3: 100.0},
        }
    )

    # Act
    res = adjust_kfz_per_hour_grid(df, 80)

    # Assert
    assert res.equals(expected)


def test_pre_fill_lags():
    """Test the output of the pre_fill_lags() function."""

    # Arrange
    dat, model_settings, date_min, changed_columns = arrange_for_lag_pred_tests()
    depvar = model_settings["depvar"]

    # Act
    dat_res = pre_fill_lags(dat.copy(deep=True), depvar, model_settings, date_min)

    dat_constant = dat.drop(columns=changed_columns)
    dat_res_constant = dat_res.drop(columns=changed_columns)
    dat_res_changed = dat_res.loc[:, changed_columns].copy(deep=True)

    # Assert
    # Every column besides the depvar and the lags has to remain unchanged
    assert dat_res_constant.equals(dat_constant), "pre_fill_lags() has modified wrong columns"

    # Check consistency within lags, i.e. lag_1 is the same as lag_2 shifted by one
    # Note: The first entry is always NaN and the last one (or two) values of the shifted columns are NaN as well
    #       -> Only compare the not NaN rows
    assert all(
        dat_res_changed["pm25_lag1"][1:-1] == dat_res_changed["pm25_lag2"].shift(periods=-1)[1:-1]
    ), "Inconsistency between lag1 and lag2"
    assert all(
        dat_res_changed["pm25_lag1"][1:-2] == dat_res_changed["pm25_lag3"].shift(periods=-2)[1:-2]
    ), "Inconsistency between lag1 and lag3"

    # Check if the avg_lag was built correctly, again ignoring the leading NaN
    assert all(
        dat_res_changed.loc[:, ["pm25_lag1", "pm25_lag2"]].sum(axis=1)[1:] / 2 == dat_res_changed["lag_avg_(1, 2)"][1:]
    ), "Inconsistency between lags and lag_avg"

    # Check that lags directly after date_min are created from real observations
    lags = [1, 2, 3]
    for lag in lags:
        date_min_lag = date_min + pd.Timedelta(lag, "hours")
        idx = dat_res_constant["date_time"] < date_min_lag
        assert all(
            dat_res_constant.loc[idx, depvar][: max(lags)]
            == dat_res_changed.shift(periods=-lag).loc[idx, f"pm25_lag{lag}"][: max(lags)]
        ), "Available real lags seem to overwritten"

    # Check remaining lags are NOT created from the observations
    for lag in [1, 2, 3]:
        date_min_lag = date_min + pd.Timedelta(lag, "hours")
        idx = dat_res_constant["date_time"] > date_min_lag
        assert not any(
            dat_res_constant.loc[idx, depvar] == dat_res_changed.loc[idx, f"pm25_lag{lag}"].shift(periods=-lag)
        ), "Real lags are used at a point, where they shouldn't accessible"

    # Check after 'date_min' there are no None values in the lag vars
    assert all(
        dat_res_changed.loc[dat_res_constant["date_time"] > date_min, :].notna()
    ), "Lag values have not been filled completely"


# __file__ = "fairqmodel/tests/test_prediction_kfz_adjusted.py"
