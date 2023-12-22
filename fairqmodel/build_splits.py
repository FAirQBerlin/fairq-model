import logging
from datetime import datetime
from logging.config import dictConfig
from typing import Optional

import pandas as pd
import pytz

from logging_config.logger_config import get_logger_config

dictConfig(get_logger_config())


def prepare_folded_input(
    dat: pd.DataFrame,
    n_cv_windows: int = 90,
    n_train_years: int = 3,
    test_cv_window_size: int = 24,
    step_size: Optional[int] = None,
    prediction_hour: Optional[int] = None,
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
    logging.info("Prepare folded data")
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

        ts_cv_dat_test = dat.loc[
            (dat.date_time > test_window_cut_min_modified) & (dat.date_time <= test_window_cut_max),
            :,
        ]

        ts_cv_dat_train = dat.loc[
            (dat.date_time > train_window_cut_min) & (dat.date_time <= test_window_cut_min_modified),
            :,
        ]

        # Store all fold details
        time_cv_folds.append(
            {
                "ts_fold_id": cv_window_idx,
                "ts_fold_max_train_date": test_window_cut_min.tz_localize(tz="UTC").tz_convert(tz="Europe/Berlin"),
                "ts_fold_max_test_date": test_window_cut_max.tz_localize(tz="UTC").tz_convert(tz="Europe/Berlin"),
                "train": ts_cv_dat_train,
                "test": ts_cv_dat_test,
            }
        )
    return time_cv_folds


def set_prediction_hour(prediction_hour: int, max_date: datetime, prediction_window_size: int) -> datetime:
    """Sets 'max_date' s.t. the predictions are performed at the selected time.

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

    new_max_date = new_max_date_berlin.astimezone(pytz.timezone("UTC")).replace(tzinfo=None)

    return new_max_date
