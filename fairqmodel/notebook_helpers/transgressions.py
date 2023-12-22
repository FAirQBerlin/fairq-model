from datetime import datetime
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from prettytable import PrettyTable

from fairqmodel.db_connect import db_connect_target, get_query
from fairqmodel.model_parameters import get_pollution_limits
from fairqmodel.prediction_kfz_adjusted import (
    make_predictions_with_adjusted_kfz_percentage,
    mark_hours_to_modify,
    prepare_dates_absolute,
    prepare_dates_current,
    select_data,
    select_day,
)
from fairqmodel.prediction_t_plus_k import get_model_settings
from fairqmodel.read_write_model_aux_functions import model_name_str


def transgression_count(
    dat: pd.DataFrame,
    station_id: str,
    depvar: str,
    tweak_value: float,
    model_id: int,
    compute_suggestions: bool = False,
) -> None:
    """Calculates on which days the allowed limit values were surpassed. Both the predictions and observations for days
     with true and false alarms are listed. For these days the highest percentage of 'kfz_per_hour' for which the
     allowed limit value is not surpassed can be calculated.

    :param dat: pd.DataFrame, Contains the predictions and observations aggregated per day
                                Columns:    station_id: at which station the observation/prediction was made
                                            date_time: for which time point the prediction was made
                                            date_time_forecast: from which time point the prediction was made
                                            pred: the predicted value
                                            <depvar>: the observed value
    :param station_id: str, Specifies which station is evaluated
    :param depvar: str, Dependent variable
    :param tweak_value: float, If provided, is used to adjust the prediction score
    :param model_id: int, ID of the model that performs the predictions
    :param compute_suggestions: bool, Specifies if for predicted transgressions a 'kfz_per_hour' percentage is suggested


    return: None
    """

    limit_value = get_pollution_limits()[depvar]

    # Create header containing meta-information
    display_header(dat, limit_value, tweak_value, station_id, depvar)

    # Select the data of provided station
    dat_station = dat.query(f"station_id == '{station_id}'")

    # Select indices of real and predicted transgressions
    real_transgr_idx = (dat_station[depvar] > limit_value).values
    pred_transgr_idx = (dat_station["pred"] + tweak_value > limit_value).values

    # True positives
    tp_idx = real_transgr_idx * pred_transgr_idx  # * performs element-wise multiplication
    create_overview(
        dat_station,
        tp_idx,
        "True Positive",
        depvar,
        limit_value,
        tweak_value,
        station_id,
        model_id,
        compute_suggestions=compute_suggestions,
    )

    # False Positives
    fp_idx = ~real_transgr_idx * pred_transgr_idx  # * performs element-wise multiplication
    create_overview(
        dat_station,
        fp_idx,
        "False Positive",
        depvar,
        limit_value,
        tweak_value,
        station_id,
        model_id,
        compute_suggestions=compute_suggestions,
    )

    # False Negatives
    fn_idx = real_transgr_idx * ~pred_transgr_idx  # * performs element-wise multiplication
    create_overview(
        dat_station,
        fn_idx,
        "False Negative",
        depvar,
        limit_value,
        tweak_value,
        station_id,
        model_id,
        compute_suggestions=False,  # Would always be 100%
    )


def create_overview(
    dat_station: pd.DataFrame,
    idx: pd.Series,
    name_metric: str,
    depvar: str,
    limit_value: int,
    tweak_value: float,
    station_id: str,
    model_id: int,
    compute_suggestions: bool = False,
) -> None:
    """Creates and displays an overview over the date on which the provided event occurred.

    :param dat_station: pd.DataFrame, Containing the predictions as well as the observations
    :param idx: pd.Series, Containing the indices at which the event occurred
    :param name_metric: str, Specifies the event
    :param depvar: str, Dependent variable
    :param limit_value: int, Allowed value for current pollutant
    :param tweak_value: float, If provided, is used to adjust the prediction score
    :param station_id: str, Specifies which station is evaluated
    :param model_id: int, ID of the model that performs the predictions
    :param compute_suggestions: bool, Specifies if for predicted transgressions a kfz_per_hour percentage is suggested

    return: None
    """
    # Access the information of each occurrence
    count = idx.sum()
    dates = [str(day) for day in dat_station.loc[idx, "date_time"].tolist()]
    pred = ["{:0.2f}".format(pred + tweak_value) for pred in dat_station.loc[idx, "pred"].tolist()]
    obs = ["{:0.2f}".format(obs) for obs in dat_station.loc[idx, depvar].tolist()]

    if compute_suggestions:
        suggestions = calculate_suggestions(dates, limit_value, tweak_value, depvar, station_id, model_id)

    # Create the overview
    if count == 0:
        print(f"No {name_metric} occurred. \n")
    else:
        table = PrettyTable()
        field_names = [name_metric, "Date", "Observation", "Prediction"]
        if compute_suggestions:
            field_names.extend(["Max. 'kfz_per_hour' Percentage"])
        table.field_names = field_names
        for x in range(count):
            row = ["", dates[x], obs[x], pred[x]]
            if compute_suggestions:
                row.append(suggestions[x])
            table.add_rows([row])
        display(table)


def display_header(dat: pd.DataFrame, limit_value: int, tweak_value: float, station_id: str, depvar: str) -> None:
    """Creates the table containing the meta-information of current evaluation"""

    table = PrettyTable()
    table.field_names = ["depvar", "Limit Value", "Tweak Value", "station_id", "Date Range"]
    date_range = f"{dat.date_time.min()} - {dat.date_time.max()}"
    table.add_rows([[depvar, limit_value, tweak_value, station_id, date_range]])
    display(table)


def find_max_kfz_percentage(
    date: str, limit_value: int, tweak_value: float, station_id: str, depvar: str, model_settings: dict
) -> str:
    """Finds the maximal percentage of 'kfz_per_hour' for which the tweaked prediction doesn't surpass the allowed
    limit value. To minimize the number of iterations a binary search is used.
    Example optimization where pred(0%) < limit_value and pred(100%) > limit value is already known:
    try 50%: pred(50%) < limit value
    try 75%: pred(75%) < limit_value
    try 87%: pred(87%) > limit value
    try 81%: pred(81%) > limit value
    try 78%: pred(78%) < limit value
    try 79%: pred(79%) > limit value
    return 78% as optimal percentage

    :param date: str, Date for which the predictions are performed
    :param limit_value: int, Allowed value for current pollutant
    :param tweak_value: float, Specifies the value that is used to modify the predictions
    :param station_id: str, Specifies which station is evaluated
    :param depvar: str, Dependent variable
    :param model_settings: dict, Containing the selected model and its meta information

    return: str, Highest percentage of 'kfz_per_hour' for which the tweaked prediction doesn't surpass the allowed
                 limit value.
    """
    percent = 0  # First percentage that is considered
    step_size = 100  # States the amount that the percentage is modified in the subsequent iteration

    # TODO need to be parameterized if the second day is evaluated
    forecast_days = 1
    day_number = 0

    # Access the selected data (incl. lag building etc.)
    date_min_absolute, date_max_absolute = prepare_dates_absolute(date, forecast_days)
    dat_all_days = select_data(date_min_absolute, date_max_absolute, model_settings, depvar, station_id)

    # Select the specified day
    date_min_current, date_max_current = prepare_dates_current(date_min_absolute, day_number)
    dat_current_day = select_day(dat_all_days, date_min_current, date_max_current, verbose=False)

    # Get the tweaked prediction for the current percentage
    pred = prediction_wrapper(dat_current_day, percent, tweak_value, model_settings)

    # If the prediction with 0% 'kfz_per_hour' is above the allowed limit value, the goal can't be reached
    if pred > limit_value:
        return "not possible"

    # Otherwise the optimal percentage is determined, as described in the doc string
    else:
        last_valid_percentage = percent
        while step_size > 1:
            direction = 1 if pred < limit_value else -1
            step_size = step_size // 2
            percent = percent + step_size * direction
            pred = prediction_wrapper(dat_current_day, percent, tweak_value, model_settings)
            if pred < limit_value:
                last_valid_percentage = percent
        return str(last_valid_percentage)


def prediction_wrapper(dat: pd.DataFrame, percentage: int, tweak_value: float, model_settings: dict) -> float:
    """Selects the hours that should be modified, performs the predictions and adds the tweak value

    :param dat: pd.DataFrame, Containing the data of the day for which the prediction is performed
    :param percentage: int, Specifies the amount of 'kfz_per_hour' that is used for the prediction
    :param tweak_value: float, Specifies the value that is used to modify the predictions
    :param model_settings: dict, Containing the selected model and its meta information

    :return: float, Tweaked prediction
    """
    # Select hours for which 'kfz_per_hour' will be modified
    hours_to_modify = mark_hours_to_modify(dat["date_time"])

    # Perform the prediction
    _, prediction_mean = make_predictions_with_adjusted_kfz_percentage(
        dat_adjusted=dat.copy(deep=True),
        percentage=percentage,
        hours_to_modify=hours_to_modify,
        date_min_current=dat.date_time.min(),
        date_max_current=dat.date_time.max(),
        model_settings=model_settings,
    )

    return prediction_mean + tweak_value


def calculate_suggestions(
    dates: List[str], limit_value: int, tweak_value: float, depvar: str, station_id: str, model_id: int
) -> List[str]:
    """Loads the selected model and calculates the suggested percentages for 'kfz_per_hour'

    :param dates: List[str], Contains the dates for which a kfz reduction is required
    :param limit_value: int, Allowed value for current pollutant
    :param tweak_value: float, Specifies the value that is used to modify the predictions
    :param depvar: str, Dependent variable
    :param station_id: str, Specifies which station is evaluated
    :param model_id: int, ID of the model that performs the predictions

    :return: Tuple[List[str], dict], Suggested percentages
    """
    # Load the model
    model_type = "all"
    query_params = {"model_type": model_name_str(model_type), "depvar": depvar}
    with db_connect_target() as db:
        available_models = db.query_dataframe(get_query("available_models"), params=query_params)

    model_settings = get_model_settings(available_models, model_id)

    # Calculate the suggestions
    suggestions = [
        find_max_kfz_percentage(date, limit_value, tweak_value, station_id, depvar, model_settings) for date in dates
    ]

    return suggestions


def plot_transgressions_per_month(
    pred_vs_obs: pd.DataFrame,
    depvar: str,
    limit_value: int,
    tweak_value: float,
    station_id: Optional[int] = None,
    save: bool = False,
) -> None:
    """Plots the transgressions per month.
    If a station_id specified, a single station is selected, otherwise all stations are considered.

    :param pred_vs_observations: pd.DataFrame, Contains the predictions and observations aggregated per day
                                Columns:    station_id: at which station the observation/prediction was made
                                            date_time: for which time point the prediction was made
                                            date_time_forecast: from which time point the prediction was made
                                            pred: the predicted value
                                            <depvar>: the observed value
    :param depvar: str, Dependent variable
    :param limit_value: int, Allowed value for current pollutant
    :param tweak_value: float, If provided, is used to adjust the prediction score
    :param station_id Optional[int], If given, only this station is examined. Otherwise all stations
    :param save: bool, Specifies if the created plot is saved

    :return: None
    """
    dat = pred_vs_obs.copy(deep=True)

    # Filter the data according to the selected station
    if station_id is not None:
        dat.query(f"station_id == '{station_id}'", inplace=True)

    # Find the days on which the allowed pollution levels were surpassed
    dat["surpassed_pred"] = dat["pred"] > limit_value
    dat["surpassed_pred_tweaked"] = (dat["pred"] + tweak_value) > limit_value
    dat["surpassed_obs"] = dat[depvar] > limit_value

    # Aggregate the transgressions on a monthly basis
    dat["month"] = pd.to_datetime(dat.date_time).dt.month
    monthly_aggregated = (
        dat.loc[:, ["month", "surpassed_pred", "surpassed_obs", "surpassed_pred_tweaked"]]
        .groupby(["month"], sort=False)
        .sum()
        .reset_index()
    )

    labels = monthly_aggregated.month
    obs = monthly_aggregated.surpassed_obs
    pred = monthly_aggregated.surpassed_pred
    pred_tweaked_ = monthly_aggregated.surpassed_pred_tweaked

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    bar1 = ax.bar(x - width, obs, width, label="Observations")
    bar2 = ax.bar(x, pred, width, label="Predictions")
    bar3 = ax.bar(x + width, pred_tweaked_, width, label="Predictions tweaked")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Number of Transgressions")
    ax.set_xlabel("Month")
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(bar1, padding=3)
    ax.bar_label(bar2, padding=3)
    ax.bar_label(bar3, padding=3)

    stations = "all stations" if station_id is None else f"station {station_id}"
    title = f"Number of Transgressions for {depvar} per month at {stations}"
    plt.suptitle(title)
    plt.grid(axis="y")

    fig.tight_layout()
    # Save the plot
    if save:
        file_name = f"transgressions_{depvar}_{stations}"
        date = datetime.now().strftime("%Y%m%d")
        plt.savefig(f"./images/transgressions_per_month/{file_name}_{date}.png")
    plt.show()
