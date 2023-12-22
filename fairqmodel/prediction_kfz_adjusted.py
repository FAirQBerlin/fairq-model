import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

from fairqmodel.data_preprocessing import cap_outliers, drop_stations_without_this_depvar, fix_column_types
from fairqmodel.db_connect import db_connect_source, get_query, send_data_clickhouse
from fairqmodel.model_parameters import get_pollution_limits, get_tweak_values
from fairqmodel.prediction_lag_adjusted import make_lag_adjusted_prediction
from fairqmodel.retrieve_data import retrieve_data
from fairqmodel.time_features import time_features
from fairqmodel.time_handling import timestamp_to_np_datetime64, to_unix


def prediction_kfz_adjusted(
    model_settings: dict,
    date_min: str,
    station_id: str,
    model_id: int,
    forecast_days: int = 2,
    write_to_db: bool = False,
    kfz_percentage: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[List[pd.DataFrame], List[bool]]:
    """Performs the prediction for a selected time window with different amounts of traffic.

    :param model_settings: dict, Containing the model and all of its required meta-information:
                                - "models": Instance of the ModelWrapper,
                                - "feature_cols": List of all features,
                                - "categorical_feature_cols": List of categorical features,
                                - "metric_feature_cols": List of categorical features,
                                - "lags": List of directly used lags,
                                - "lags_avg": List of Lags used in lag average,
                                - "depvar": Dependent variable,
                                - "max_training_date_model_1",
                                - "max_training_date_model_2",
    :param date_min: str, Date at which the forecasts starts, Berlin Time
    :param station_id: str, Specifies the station at which the predictions are made
    :param model_id: int, Specifies the model that performs the predictions
    :param forecast_days: int, Number of days to predict into the future
    :param write_to_db: bool, Specifies if the predictions are written to the DB
    :param kfz_percentage: Optional[int], If provided, the prediction is performed with this percentage
    :param verbose: bool, Specifies if progress is logged

    :return: List[pd.DataFrame], One DataFrame per evaluated percentage. Each DataFrame contains:
                        - "date_time_forecast",
                        - "date_time",
                        - "station_id",
                        - "depvar",
                        - "pred",

    :return: List[bool], Specifies the hours for which 'kfz_per_hour' was modified
    """

    # Access the model settings
    depvar = model_settings["depvar"]

    # Access threshold value and tweak value per depvar
    limit_value = int(get_pollution_limits()[depvar])
    tweak_value = float(get_tweak_values()[depvar])

    # Select and access the data
    date_min_absolute, date_max_absolute = prepare_dates_absolute(date_min, forecast_days)
    dat_all_days = select_data(date_min_absolute, date_max_absolute, model_settings, depvar, station_id)

    all_results: List[pd.DataFrame] = []

    for day_number in range(forecast_days):
        # Select time frame
        date_min_current, date_max_current = prepare_dates_current(date_min_absolute, day_number)
        dat_current_day = select_day(dat_all_days, date_min_current, date_max_current, verbose)

        # Select hours for which 'kfz_per_hour' will be modified
        hours_to_modify = mark_hours_to_modify(dat_current_day["date_time"])

        if dat_current_day.shape[0] != 24:
            hours = dat_current_day.shape[0]
            day = date_min_current.date()
            logging.warning(f"Prediction for the '{day}' was aborted, since only data for {hours} hours is available.")
            return all_results, list()

        # DataFrame for results
        dat_results = pd.DataFrame(
            {
                "station_id": [station_id],
                "model_id": [model_id],
                "date": [date_min_current.date()],
                "date_time_forecast": [to_unix(date_min_absolute)],
                "pollutant_limit": [limit_value],
                "tweak_value": [tweak_value],
            }
        )

        # Perform the adjusted predictions for the selected percentages of 'kfz_per_hour'
        percentages = np.arange(0, 101, 10).tolist() if kfz_percentage is None else [kfz_percentage]

        for percentage in percentages:
            # Perform predictions with adjusted 'kfz_per_hour'
            dat_prediction, prediction_mean = make_predictions_with_adjusted_kfz_percentage(
                dat_adjusted=dat_current_day.copy(deep=True),
                percentage=percentage,
                hours_to_modify=hours_to_modify,
                date_min_current=dat_current_day.date_time.min().tz_localize(tz="UTC").tz_convert(tz="Europe/Berlin"),
                date_max_current=dat_current_day.date_time.max().tz_localize(tz="UTC").tz_convert(tz="Europe/Berlin"),
                model_settings=model_settings,
            )

            # Store the daily avg in for the current percentage in the DataFrame
            dat_results[f"pct_kfz_{percentage}"] = prediction_mean
            # Append all predictions to the result list
            all_results.append(dat_prediction.copy(deep=True))

        # Write predictions to DB
        if write_to_db:
            assert kfz_percentage is None, "Results for manually selected percentage can not be written to DB."

            # Send the DataFrame to DB
            send_data_clickhouse(df=dat_results, table_name="model_predictions_thresholds", mode="replace")

    return all_results, hours_to_modify.tolist()


def pre_fill_lags(
    dat: pd.DataFrame, depvar: str, model_settings: dict, date_min_absolute: pd.Timestamp
) -> pd.DataFrame:
    """The kfz_adjusted_prediction is performed for several days into the future.
    The kfz reductions however are evaluated on a daily basis.
    If e.g. the prediction is made for the upcoming two days, some lags of the second day are
    build from the first day. Due to the per-day calculation, these lags usually wouldn't be available.
    To overcome this problem, this function performs one forecast over the entire selected time horizon (with 100% kfz)
    and pre-fills the lag columns based on these prediction.

    :param dat: pd.DataFrame, Containing all data
    :param depvar: str, Dependent variable
    :param model_settings: dict, Containing the model and its corresponding information
    :param date_min_absolute: pd.Timestamp, Date from which the forecast is performed, in Berlin Time

    :return: pd.DataFrame, Same DataFrame as input, but missing lags have been added
    """

    dat.reset_index(drop=True, inplace=True)

    observations_backup = dat.loc[:, depvar].copy(deep=True)

    # Remove all observations after 'min_date'
    # NOTE: 'date_min_absolute' is in Berlin Time.
    # Since the 'date_time' column is tz naive but the dates are in UTC, 'date_min_absolute' is casted to UTC
    future_idx = dat["date_time"] > timestamp_to_np_datetime64(date_min_absolute)

    dat.loc[future_idx, depvar] = None

    # Build time features and fix dtypes
    dat = time_features(dat, depvar=depvar, lags_actual=model_settings["lags"], lags_avg=model_settings["lags_avg"])
    dat = fix_column_types(
        dat,
        categorical_feature_cols=model_settings["categorical_feature_cols"],
        metric_feature_cols=model_settings["metric_feature_cols"],
    )

    # Perform the forecast for this part of the data (possibly for several days)
    forecast = make_forecast(
        date_min=date_min_absolute,
        date_max=dat.date_time.max().tz_localize(tz="UTC").tz_convert(tz="Europe/Berlin"),
        dat=dat.loc[future_idx, :],
        model_settings=model_settings,
    )

    # Fill depvar column after 'date_min' with predicted values
    dat.loc[future_idx, depvar] = forecast["pred"].values

    # Update the lags, now including all future time steps
    dat = time_features(dat, depvar=depvar, lags_actual=model_settings["lags"], lags_avg=model_settings["lags_avg"])

    # Change the temporarily overwritten observation values back to their original value
    dat.loc[:, depvar] = observations_backup

    dat = fix_column_types(dat, model_settings["categorical_feature_cols"], model_settings["metric_feature_cols"])

    return dat


def make_predictions_with_adjusted_kfz_percentage(
    dat_adjusted: pd.DataFrame,
    percentage: int,
    hours_to_modify: pd.DataFrame,
    date_min_current,
    date_max_current,
    model_settings: dict,
) -> Tuple[pd.DataFrame, float]:
    """Performs the prediction with an adjusted amount of traffic."""

    # Adjust the 'kfz_per_hour' by the selected amount
    dat_adjusted.loc[hours_to_modify, "kfz_per_hour"] = (
        dat_adjusted.loc[hours_to_modify, "kfz_per_hour"] * percentage / 100
    )

    # Perform the forecast with the adjusted data
    dat_prediction = make_forecast(date_min_current, date_max_current, dat_adjusted, model_settings)

    # Store the daily avg in for the current percentage in the DataFrame
    prediction_mean = dat_prediction["pred"].mean()

    return dat_prediction, prediction_mean


def make_forecast(
    date_min: pd.Timestamp, date_max: pd.Timestamp, dat: pd.DataFrame, model_settings: dict
) -> pd.DataFrame:
    """Makes the forecast for the provided data and time window.

    :param date_min: pd.Timestamp, Minimal date of the forecast, in Berlin Time
    :param date_max: pd.Timestamp, Maximal date of the forecast, in Berlin Time
    :param dat: pd.DataFrame, Containing the data for which the forecast is made
    :param model_settings: dict, Containing information about the used model

    :return: pd.DataFrame
    """

    data_dict = {
        "ts_fold_id": 1,
        "ts_fold_max_train_date": date_min,
        "ts_fold_max_test_date": date_max,
        "train": pd.DataFrame({}),  # No training here
        "test": dat.copy(deep=True),  # Data for prediction, copy to avoid overwriting
    }

    # Perform the forecast
    dat_prediction, _ = make_lag_adjusted_prediction(
        data_dict,
        models=model_settings["models"],
        feature_cols=model_settings["feature_cols"],
        depvar=model_settings["depvar"],
        lags_actual=model_settings["lags"],
        lags_avg=model_settings["lags_avg"],
        calc_metrics=False,
    )

    # Reorder columns
    dat_prediction = dat_prediction[["date_time_forecast", "date_time", "station_id", model_settings["depvar"], "pred"]]

    return dat_prediction


def adjust_kfz_per_hour_grid(dat: pd.DataFrame, percentage: int = 100) -> pd.DataFrame:
    """
    adjust the kfz per hour for grid predictions

    :param dat: pd.DataFrame with a column "kfz_per_hour"
    :param percentage: int. percentage to adjust the kfz_per_hour between 5am and 9pm. 100 means no adjustment.

    :return copy of dat (input pd.DataFrame) with the adjusted column kfz_per_hour
    """

    if percentage != 100:
        logging.info(f"Adjusting kfz_per_hour with percentage = {percentage}\n")

    dat_adjusted = dat.copy()
    hours_to_modify = mark_hours_to_modify(dat_adjusted["date_time"])

    # Adjust the 'kfz_per_hour' by the selected amount
    dat_adjusted.loc[hours_to_modify, "kfz_per_hour"] = (
        dat_adjusted.loc[hours_to_modify, "kfz_per_hour"] * percentage / 100
    )

    return dat_adjusted


def mark_hours_to_modify(date_time_column: pd.Series) -> pd.Series:
    """
    Mark hours for which the traffic may be modified
    :param date_time_column: Date time column of the data for which the prediction will be performed
    :return: Series of Booleans, with the same length as the input. Each Bool signals if the corresponding hour is
             eligible for traffic reduction.
    """
    berlin_tz = pytz.timezone("Europe/Berlin")
    datetimes_in_berlin_time = date_time_column.dt.tz_localize("UTC").dt.tz_convert(berlin_tz)
    selected_hours = np.arange(5, 22).tolist()  # Adjustments are performed between 5am and 9pm
    hours_to_modify = [dt.hour in selected_hours for dt in datetimes_in_berlin_time]
    return pd.Series(hours_to_modify, name="date_time", dtype=bool)


def check_date_min(date_min: pd.Timestamp) -> pd.Timestamp:
    """Checks if the selected 'date_min' is available.
    If yes it remains unchanged, if no, the most recent available date is returned.

    :param date_min: pd.Timestamp, Selected 'date_min' Berlin time

    :return: pd.Timestamp, Available 'date_min' in Berlin Time
    """
    with db_connect_source() as db:
        most_recent_observation_date = (
            pd.Timestamp(db.query_dataframe(get_query("most_recent_observation")).values[0][0])
            .tz_localize(tz="UTC")  # Entries in the DB are in UTC, but time zone naive
            .tz_convert(tz="Europe/Berlin")
        )

    if date_min - pd.Timedelta(1, "hours") > most_recent_observation_date:
        logging.warning(f"Forecast for selected min_date {date_min.date()} can't be performed")
        logging.info(
            "Forecast is performed with most recent available min_date of {}".format(
                most_recent_observation_date.date()
            )
        )
        date_min = most_recent_observation_date + pd.Timedelta(1, "hours")

    return date_min


def prepare_dates_absolute(date_min: str, forecast_days: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Selects valid dates according to the provided parameters

    :param date_min: str, Earliest date for which the predictions are performed, Berlin Time
    :param forecast_days: int, Specifies how many days into the future the predictions are made

    :return: Tuple[pd.Timestamp, pd.Timestamp],
             Minimal and maximal dates for the entire subsequent prediction, both in Berlin Time
    """
    date_min_absolute = pd.Timestamp(date_min, tz="Europe/Berlin")
    date_max_absolute = date_min_absolute + pd.Timedelta(forecast_days, "days")

    # Check if selected 'date_min' is available
    date_min_absolute = check_date_min(date_min_absolute)

    return date_min_absolute, date_max_absolute


def prepare_dates_current(date_min_absolute: pd.Timestamp, day_number: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Selects the current date relatively to the given absolute date.
    'date_min_absolute' is in Berlin Time and consequently both returned dates are in Berlin Time too.

    :param date_min_absolute: pd.Timestamp, Earliest date of the entire prediction window, Berlin Time
    :param day_number: int, Specifies which day is selected from the DataFrame

    :return: Tuple[pd.Timestamp, pd.Timestamp], Minimal and maximal date of the selected day, Berlin Time
    """
    date_min_current = date_min_absolute + pd.Timedelta(day_number, "days")
    date_max_current = date_min_current + pd.Timedelta(1, "days")

    return date_min_current, date_max_current


def select_data(
    date_min_absolute: pd.Timestamp, date_max_absolute: pd.Timestamp, model_settings: dict, depvar: str, station_id: str
) -> pd.DataFrame:
    """Selects the data for the specified range and builds the required lags.

    :param date_min_absolute: pd.Timestamp, Earliest date for which the predictions are performed, in Berlin Time
    :param date_max_absolute: pd.Timestamp, Latest date up to which the predictions are performed, in Berlin Time
    :param model_settings: dict, Containing the selected model and its meta information
    :param depvar: str, Dependent variable
    :param station_id: str, Specifies which station is evaluated

    :return: pd.DataFrame, Pre-processed data
    """
    # Retrieve selected data
    max_lag = max(model_settings["lags"] + model_settings["lags_avg"])
    dat_all_days = retrieve_data(
        mode="stations",
        date_time_max=date_max_absolute,
        date_time_min=date_min_absolute - pd.Timedelta(max_lag + 1, "hour"),
    )
    dat_all_days = drop_stations_without_this_depvar(dat_all_days, depvar)

    # Remove outliers
    dat_all_days = cap_outliers(dat_all_days)

    # Preprocess the data
    dat_all_days.query(f"station_id == '{station_id}'", inplace=True)

    # Pre-fill lag columns based on normal predictions (i.e. with 100% 'kfz_per_hour'):
    dat_all_days = pre_fill_lags(dat_all_days, depvar, model_settings, date_min_absolute)

    # Cut off dates prior to date min (that were required to build lags)
    dat_all_days.query(f"date_time >= '{timestamp_to_np_datetime64(date_min_absolute)}'", inplace=True)

    return dat_all_days


def select_day(
    dat_all_days: pd.DataFrame,
    date_min_current: pd.Timestamp,
    date_max_current: pd.Timestamp,
    verbose: bool,
) -> pd.DataFrame:
    """Selects a single day from the larger DataFrame

    :param dat_all_days: pd.DataFame, Containing the data of all days for which the prediction is made
    :param date_min_current: pd.Timestamp, Start of the selected day, in Berlin Time
    :param date_max_current: pd.Timestamp, End of the selected day, in Berlin Time
    :param verbose: bool, Specifies if the selected day is logged

    :return: pd.DataFrame, Containing the data of the selected day
    """
    if verbose:
        logging.info(f"Now predicting day {date_min_current.date()}")

    # Select relevant data
    dat_current_day = dat_all_days.copy(deep=True)

    # Select time points from 1 am from the current day until 0 am of the next day
    # Example: For the '2022-10-09' the time points ['2022-10-09 01:00:00', '2022-10-10 00:00:00']
    dat_current_day.query(
        "date_time > '{}' and date_time <= '{}'".format(
            timestamp_to_np_datetime64(date_min_current), timestamp_to_np_datetime64(date_max_current)
        ),
        inplace=True,
    )
    dat_current_day.reset_index(drop=True, inplace=True)

    return dat_current_day
