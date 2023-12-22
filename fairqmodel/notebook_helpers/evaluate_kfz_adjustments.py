from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fairqmodel.model_parameters import get_pollution_limits, get_tweak_values
from fairqmodel.prediction_kfz_adjusted import mark_hours_to_modify, prediction_kfz_adjusted


def plot_kfz_prediction(
    depvar: str,
    all_results: List[pd.DataFrame],
    date_forecast: str,
    modified_hours: List[bool],
    station_id: str,
    save: bool = False,
) -> None:
    """Plot the predictions with different percentages of 'kfz_per_hour'

    :param depvar: str, Dependent variable
    :param all_results: List[pd.DataFrame], Contains the predictions. One DataFrame per day and percentage
    :param date_forecast: str, Specifies the date at which the forecast is performed
    :param modified_hours: List[bool], Specifies the hours for that have been modified
    :param station_id: str, Id of the station for which the predictions are made
    :param save: bool, Specifies if plot is saved

    :return: None
    """

    # Access target value and tweak value per depvar
    limit_value = get_pollution_limits()[depvar]
    tweak_value = get_tweak_values()[depvar]

    pred_per_day = 11  # 0, 10, 20, ..,  100 percent
    colors = ["green", "blue", "orange"]

    # Number of days for which a the forecast was created (migh be smaller than the selected number)
    actual_forecast_days = len(all_results) // pred_per_day

    # Init plot
    plt.figure(figsize=(5 * actual_forecast_days, 6))

    # Settings x-axis
    x_ticks = np.arange(0, actual_forecast_days * 24)
    x_labels = (np.arange(1, 24).tolist() + [0]) * actual_forecast_days
    plt.xticks(labels=x_labels, ticks=x_ticks)
    plt.xlim(0, actual_forecast_days * 24 - 1)
    plt.xlabel("Hours")

    # Emphasize hours that have been modified
    locs, labels = plt.xticks()
    all_hours = np.arange(0, 23).tolist()
    modified_idx = [x for x in all_hours if modified_hours[x]]
    for idx in modified_idx:
        for day in range(0, actual_forecast_days):
            labels[idx + day * 24].set_color("purple")

    # Access the observed values for all days
    obs = []
    for idx, i in enumerate(range(0, len(all_results), pred_per_day)):
        current_obs = all_results[i][depvar]
        obs.extend(current_obs)
        # Plot the observed daily mean (if any observations are available)
        if current_obs.notna().any():
            plt.hlines(
                y=current_obs.mean(),
                xmin=idx * 24,
                xmax=(idx + 1) * 24,
                linewidth=1,
                color="black",
            )

    # Max value, required for nice plot scaling
    max_value = max([x for x in obs if x is not None])
    max_value = max([max_value, limit_value])

    # Plot observations
    plt.plot(x_ticks, obs, label="Observation", color="black")

    # Plot the allowed limit value
    plt.hlines(y=limit_value, xmin=0, xmax=actual_forecast_days * 24, color="red", label="Allowed Limit")

    # Access and plot predictions for different percentages of kfz
    for idx_out, j in enumerate(range(0, pred_per_day, 5)):  # Loop over different percentages (only for 0, 50 and 100%)
        pred = []
        for idx_in, i in enumerate(range(j, len(all_results), pred_per_day)):  # Loop over all days
            pred.extend(all_results[i]["pred"])
            # Plot daily prediction average
            plt.hlines(
                y=all_results[i]["pred"].mean() + tweak_value,
                xmin=idx_in * 24,
                xmax=(idx_in + 1) * 24,
                linewidth=0.5,
                color=colors[idx_out],
            )
        # Plot predictions for all days
        plt.plot(x_ticks, np.array(pred) + tweak_value, label=f"Pred. {j*10} % kfz", color=colors[idx_out])

        if max(pred) > max_value:
            max_value = max(pred)

    # Settings y-axis
    plt.ylim(0, (max_value + tweak_value) * 1.05)
    plt.ylabel("Pollutant level")

    # General Settings
    plt.grid()
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title(
        "Forecast for {} starting at the {} for the upcoming {} days at station {}".format(
            depvar, date_forecast, actual_forecast_days, station_id
        )
    )

    # Plot beginning of new days
    for x in np.arange(0, actual_forecast_days * 24, 24):
        plt.vlines(x=x, ymax=(max_value + tweak_value) * 1.05, ymin=0, colors="black", linestyles="dashed")

    if save:
        title = f"{depvar}_{date_forecast}_{actual_forecast_days}_days_station_{station_id}"
        date = datetime.now().strftime("%Y%m%d")
        plt.savefig(f"./images/kfz_predictions/{depvar}/{title}_{date}.png")

    plt.show()


def calculate_kfz_contribution(
    model_settings: dict, station_id: str, model_id: int, date_min: str, n_days: int
) -> None:
    """Performs a prediction with 0% 'kfz_per_hour' and one with 100% 'kfz_per_hour' for every selected day.
    For each day the amount (in percent) that is contributed by the 'kfz_per_hour' is calculated. Then the average
    over all daily contributions is built.

    :param model_settings: dict, Containing the trained model and its meta information
    :param station_id: str, Specifies the station at which the predictions are made
    :param model_id: Specifies the model used for prediction
    :param date_min: str, Start date of the evaluation
    :param n_days: int, Specifies the amount of days that are used

    :return: None
    """
    date_min = pd.Timestamp(date_min, tz="UTC")
    date_max = date_min + pd.Timedelta(n_days, "days")

    all_days: np.ndarray = np.arange(date_min, date_max, pd.Timedelta(1, "days"))

    predictions_kfz_0 = []
    predictions_kfz_100 = []

    for day in all_days:
        current_day = str(np.datetime_as_string(day, unit="D"))

        # Predict current day with 0% 'kfz_per_hour'
        preds_0 = prediction_kfz_adjusted_relevant_hours(0, model_settings, current_day, station_id, model_id)
        predictions_kfz_0.extend(preds_0)

        # Predict current day with 100% 'kfz_per_hour'
        preds_100 = prediction_kfz_adjusted_relevant_hours(100, model_settings, current_day, station_id, model_id)
        predictions_kfz_100.extend(preds_100)

    zero_pred = np.array(predictions_kfz_0)
    full_pred = np.array(predictions_kfz_100)

    # Mean of the daily contribution of 'kzf_per_hour' to the total pollution level
    kfz_contribution = 100 * (1 - (zero_pred / full_pred).mean())
    print(
        "'kfz_per_hour' contributes {:0.2f}% to the total {} level.".format(kfz_contribution, model_settings["depvar"])
    )


def prediction_kfz_adjusted_relevant_hours(
    kfz_percentage: int, model_settings: dict, current_day: str, station_id: str, model_id: int
) -> List:
    """Performs the kfz_adjusted_prediction for a given percentage and day. Only the predictions for hours that are
    eligible for kfz_adjustments are returned.

    :param kfz_percentage: int, Specifies the amount of "kfz_per_hour" that is used for the prediction
    :param model_settings: dict, Containing the trained model and its meta information
    :param current_day: str, Date for which the prediction is performed
    :param station_id: str, Specifies the station at which the predictions are made
    :param model_id: Specifies the model used for the prediction

    """
    all_results, _ = prediction_kfz_adjusted(
        model_settings,
        current_day,
        station_id,
        model_id,
        forecast_days=1,
        kfz_percentage=kfz_percentage,
        write_to_db=False,
        verbose=False,
    )
    res = all_results[0]
    hours_modified = mark_hours_to_modify(res.date_time)  # Consider only hours that have been modified
    preds = res.loc[hours_modified, "pred"].values.tolist()

    return preds
