"""Build time series cross-validation splits for model training and evaluation."""

from datetime import datetime

import pandas as pd
import pytz
from loguru import logger


def prepare_folded_input(
    dat: pd.DataFrame,
    n_cv_windows: int = 90,
    n_train_years: int = 3,
    test_cv_window_size: int = 24,
    step_size: int | None = None,
    prediction_hour: int | None = None,
    include_current_time_point: bool = False,
) -> list[dict]:
    """Prepare the input for the time series cross validation.

    Args:
        dat: pd.DataFrame, data
        n_cv_windows: int, number of cross validation windows
        n_train_years: int, number of years for training
        test_cv_window_size: int, size of the test cross validation window
        step_size: int, number of hours two adjacent windows are shifted against each other.
        prediction_hour: int, if not None, specifies the last hour of training,
                         i.e. the time point where the prediction is made
        include_current_time_point: bool, Specifies if a prediction for the current time point (k=0) is made.
                                           Default is False
    Returns:
        List[dict], each dict contains the train and test data

    """
    logger.info("Prepare folded data")
    days_per_year = 365
    hours_per_day = 24
    train_window_size = n_train_years * days_per_year * hours_per_day

    if step_size is None:
        step_size = test_cv_window_size

    assert "date_time" in dat.columns, "DataFrame has to contain a date_time column!"

    max_date = dat.date_time.max()  # Dates from the DB are in UTC but tz naive

    if prediction_hour is not None:
        max_date = set_prediction_hour(prediction_hour, max_date, test_cv_window_size)

    time_cv_folds = []
    for cv_window_idx in range(1, n_cv_windows + 1):
        # how far in the past do we have to go to do the splitting
        # Note: td_hours specifies how much each window is shifted against the previous one
        #       If step_size < test_cv_window_size this would lead to
        #       test_window sizes smaller than specified in test_cv_window_size for the most recent date times.
        #       To avoid this behavior the most recent date_time_forecast is at least one test_window_size in the past.
        td_hours = (cv_window_idx - 1) * step_size + test_cv_window_size

        # get the time cuts for the train and test data
        test_window_cut_min = max_date - pd.Timedelta(f"{td_hours} hours")
        test_window_cut_max = test_window_cut_min + pd.Timedelta(f"{test_cv_window_size} hours")
        train_window_cut_min = test_window_cut_min - pd.Timedelta(f"{train_window_size} hours")

        # filter the data
        test_window_cut_min_modified = test_window_cut_min
        if include_current_time_point:
            test_window_cut_min_modified -= pd.Timedelta("1 hours")

        # Store all fold details
        time_cv_folds.append(
            {
                "ts_fold_id": cv_window_idx,
                "ts_fold_max_train_date": convert_to_local_time(test_window_cut_min),
                "ts_fold_max_test_date": convert_to_local_time(test_window_cut_max),
                "train_window_cut_min": train_window_cut_min,
                "test_window_cut_min_modified": test_window_cut_min_modified,
                "test_window_cut_max": test_window_cut_max,
            },
        )
    return time_cv_folds


def convert_to_local_time(time_stamp):
    """Convert a UTC timestamp to Europe/Berlin local time."""
    return time_stamp.tz_convert(
        tz="Europe/Berlin",
    )


def slice_data_frame(
    dat: pd.DataFrame,
    lower_bound: datetime,
    upper_bound: datetime,
    slice_column: str = "date_time",
) -> pd.DataFrame:
    """Slices the DataFrame between the lower and upper bound.

    :param dat: pd.DataFrame, Data to slice
    :param lower_bound: datetime, Lower bound of the slice
    :param upper_bound: datetime, Upper bound of the slice
    :param slice_column: str, Column to slice the DataFrame on

    :return: pd.DataFrame, Sliced DataFrame
    """
    return dat.loc[
        (dat[slice_column] > lower_bound) & (dat[slice_column] <= upper_bound),
        :,
    ]


def set_prediction_hour(prediction_hour: int, max_date: datetime, prediction_window_size: int) -> datetime:
    """Set 'max_date' s.t. the predictions are performed at the selected time.

    :param prediction_hour: int, Hour of the day, when the prediction should be performed, Berlin Time
    :param max_date: datetime, Maximal available date, in UTC but tz naive
    :param prediction_window_size: int, Number of hours to predict into the future for each prediction

    :return:datetime, New 'max_date' with selected 'prediction_hour', in UTC but tz naive
    """
    max_date_berlin = pytz.timezone("UTC").localize(max_date).astimezone(pytz.timezone("Europe/Berlin"))
    first_pred_date = max_date_berlin - pd.Timedelta(f"{prediction_window_size} hours")

    if first_pred_date.hour > prediction_hour:
        # Floor down to correct hour
        new_max_date_berlin = max_date_berlin - pd.Timedelta(first_pred_date.hour - prediction_hour, "hours")
    elif first_pred_date.hour < prediction_hour:
        # Increase to correct hour and decrease by one day
        new_max_date_berlin = max_date_berlin - pd.Timedelta(24 + (first_pred_date.hour - prediction_hour), "hours")
    else:
        # Prediction is already performed at correct hour
        new_max_date_berlin = max_date_berlin

    return new_max_date_berlin.astimezone(pytz.timezone("UTC")).replace(tzinfo=None)
