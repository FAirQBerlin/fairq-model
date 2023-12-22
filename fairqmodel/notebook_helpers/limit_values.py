import logging
from logging.config import dictConfig
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from fairqmodel.data_preprocessing import fix_column_types
from fairqmodel.db_connect import db_connect_target, get_query
from logging_config.logger_config import get_logger_config

dictConfig(get_logger_config())


def preprocess_data_for_limit_values(
    depvar: str, model_id: int, forecast_horizon_days: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads and pre-processes the observations and predictions for a given model.
    The data is used to identify whether allowed pollution levels were exceeded.

    :param depvar: str, Dependent variable
    :param model_id: int, ID of the model that performs the predictions
    :param forecast_horizon_days: int, Specifies the day of the forecast that is evaluated

    :return: Tuple[pd.DataFrame, pd.DataFrame], First DataFrame containing all information,
                                                Second DataFrame containing predictions/observations
                                                aggregated on daily basis.
    """

    # Retrieve data from db
    with db_connect_target() as db:
        dat = db.query_dataframe(get_query("limit_values_analysis"), params={"model_id": model_id})

    # Remove two not selected depvars
    dat.drop(columns=list({"no2", "pm25", "pm10"}.difference({depvar})), inplace=True)

    # Fix column type
    dat = fix_column_types(dat, ["station_id"], [depvar])

    # select forecast range
    assert forecast_horizon_days <= 2, f"There are no predictions for {forecast_horizon_days} days into the future."
    # TODO Modify this assert as soon as a final model was used to perform predictions for the year of interest.
    # Move it to query
    low = pd.Timedelta(24 * (forecast_horizon_days - 1), "hour")
    high = pd.Timedelta(24 * forecast_horizon_days, "hour")
    dat = dat[(dat.date_time - dat.date_time_forecast <= high) & (dat.date_time - dat.date_time_forecast > low)]

    # Aggregate data on daily basis
    pred_vs_observation, dat = aggregate_data_daily(dat)

    # Check for duplicates
    if sum(pred_vs_observation.duplicated(subset=["station_id", "date_time"])) > 0:
        logging.warning("The data contains duplicates.")

    # If values are missing, remove them
    missing_values_count = pred_vs_observation[depvar].isna().sum()
    if missing_values_count > 0:
        pred_vs_observation = pred_vs_observation.dropna(subset=[depvar])
        logging.info(f"{missing_values_count} missing values have been removed.")

    return dat, pred_vs_observation


def aggregate_data_daily(dat: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Computes the mean prediction and observation per day.
    The dates in the DataFrame as in UTC but tz naive. They are cast to Berlin Time before processing.
    NOTE: Includes 0 am to the previous day (as requested).

    :param dat: pd.DataFrame, Containing unaggregated data

    :return: pd.DataFrame, Containing the aggregated date
    :return: pd.DataFrame, Original DataFrame but 'date_time' column is in Berlin Time (and as before tz naive)
    """
    # Interpret dates in DataFrame as UTC and convert to Berlin Time
    dat.loc[:, "date_time"] = dat["date_time"].dt.tz_localize("UTC").dt.tz_convert(tz="Europe/Berlin")

    # Aggregate on the 'date_time' column (which is currently in Berlin Time)
    pred_vs_observation = dat.groupby("station_id").resample("D", on="date_time", closed="right").mean().reset_index()
    pred_vs_observation["date_time"] = pred_vs_observation["date_time"].dt.strftime("%Y-%m-%d")
    dat.loc[:, "date_time"] = dat["date_time"].dt.tz_localize(None)
    return pred_vs_observation, dat


def calc_conf_matrix(
    depvar: str, pred_vs_observation: pd.DataFrame, limit_value: int, tweak_value: float = 0.0, plot: bool = False
) -> Tuple[confusion_matrix, confusion_matrix]:
    """Calculates both the absolute and the normalized confusion matrices for the given predictions,
        limit values and tweak values. If selected, the CMs can be plotted.

    :param depvar: str, Dependent variable
    :param pred_vs_observations: pd.DataFrame, Contains the predictions and observations aggregated per day
                                    Columns:    station_id: at which station the observation/prediction was made
                                                date_time: for which time point the prediction was made
                                                date_time_forecast: from which time point the prediction was made
                                                pred: the predicted value
                                                <depvar>: the observed value
    :param limit_value: int, Allowed value for current pollutant
    :param tweak_value: float, Specifies the value that is used to modify the predictions
    :param plot: bool, False, Specifies if confusion matrix is plotted

    :return: Tuple[confusion_matrix, confusion_matrix]
    """

    cm = confusion_matrix(
        pred_vs_observation[depvar] > limit_value,
        pred_vs_observation["pred"] + tweak_value > limit_value,
        labels=[True, False],
    )

    cm_normalized = confusion_matrix(
        pred_vs_observation[depvar] > limit_value,
        pred_vs_observation["pred"] + tweak_value > limit_value,
        labels=[True, False],
        normalize="true",
    )

    if plot:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Above limit", "Below limit"])
        disp.plot()
        plt.suptitle(f"Predictions for {depvar} \n  limit value: {limit_value}, tweak value: {tweak_value}")
        plt.show()

    return cm, cm_normalized


def plot_scores_per_tweak_value(
    depvar: str, pred_vs_obs: pd.DataFrame, limit_value: int, min_val: int, max_val: int
) -> None:
    """Plots the prediction metrics in dependence of the tweak value.

    :param depvar: str, Dependent variable
    :param pred_vs_observations: pd.DataFrame, Contains the predictions and observations aggregated per day
                                Columns:    station_id: at which station the observation/prediction was made
                                            date_time: for which time point the prediction was made
                                            date_time_forecast: from which time point the prediction was made
                                            pred: the predicted value
                                            <depvar>: the observed value
    :param limit_value: int, Allowed value for current pollutant
    :param min_val: int, Minimal value considered as tweak value
    :param max_val: int, Maximal value considered as tweak value

    :return: None
    """

    tweak_values = np.arange(min_val, max_val + 1, 0.1)
    tn_list = []
    fp_list = []
    fn_list = []
    tp_list = []
    f1_list = []

    for tw in tweak_values:
        tn, fp, fn, tp, f1 = calc_classification_metric(depvar, pred_vs_obs, limit_value, tw)
        tn_list.append(tn)
        fp_list.append(fp)
        fn_list.append(fn)
        tp_list.append(tp)
        f1_list.append(f1)

    plt.figure(figsize=(12, 6))
    plt.plot(tweak_values, tn_list, label="tn_rate", linewidth=2.0, color="blue")
    plt.plot(tweak_values, fp_list, label="fp_rate", linewidth=2.0, color="red")
    plt.plot(tweak_values, fn_list, label="fn_rate", linewidth=2.0, color="purple")
    plt.plot(tweak_values, tp_list, label="tp_rate", linewidth=2.0, color="green")
    plt.plot(tweak_values, f1_list, label="f1_score", linewidth=2.0, color="orange")
    plt.yticks(np.arange(0, 1.001, 0.1))
    plt.xticks(np.arange(tweak_values.min(), tweak_values.max() + 1, 1))
    plt.xlabel("tweak_value")
    plt.ylabel("score")
    plt.vlines(x=0, ymax=1.1, ymin=-0.1, color="black")

    plt.ylim(-0.1, 1.1)
    plt.xlim(tweak_values.min(), tweak_values.max())
    plt.legend()
    plt.grid()
    plt.title(f"Metrics per tweak value for {depvar}")
    plt.show()


def find_tweak_value(
    depvar: str,
    pred_vs_obs: pd.DataFrame,
    limit_value: int,
    min_val: int = -20,
    max_val: int = 20,
    step_size: float = 1.0,
) -> float:
    """Finds the smallest tweak value for which the fp-rate is not larger than the fn-rate.
    NOTE: Assuming that both the fp-rate and fn-rate are monotonic w.r.t. to the tweak value
          this is the only point where fp-rate and fn-rate are (nearly) equal.

    :param depvar: str, Dependent variable
    :param pred_vs_observations: pd.DataFrame, Contains the predictions and observations aggregated per day
                                Columns:    station_id: at which station the observation/prediction was made
                                            date_time: for which time point the prediction was made
                                            date_time_forecast: from which time point the prediction was made
                                            pred: the predicted value
                                            <depvar>: the observed value
    :param limit_value: int, Allowed value for current pollutant
    :param min_val: int, Minimal value considered as tweak value
    :param max_val: int, Maximal value considered as tweak value
    :param step_size: float, Specifies how fine-grained the search is

    :return: float, The optimal tweak value
    """
    tweak_values = np.arange(min_val, max_val + 1, step_size)

    for tw in tweak_values:
        res = calc_classification_metric(depvar, pred_vs_obs, limit_value, tw)
        fp = res[1]
        fn = res[2]
        if fp <= fn:
            break

    return tw


def calc_classification_metric(
    depvar: str,
    pred_vs_obs: pd.DataFrame,
    limit_value: int,
    tweak_value: float,
) -> Tuple[float, float, float, float, float]:
    """Calculates the metrics for the binary classification whether a prediction exceeds the allowed limit_value.

    :param depvar: str, Dependent variable
    :param pred_vs_observations: pd.DataFrame, Contains the predictions and observations aggregated per day
                                Columns:    station_id: at which station the observation/prediction was made
                                            date_time: for which time point the prediction was made
                                            date_time_forecast: from which time point the prediction was made
                                            pred: the predicted value
                                            <depvar>: the observed value
    :param limit_value: int, Allowed value for current pollutant
    :param tweak_value: float, Specifies the value that is used to modify the predictions

    :return: Tuple[float, float, float, float, float]
    """
    _, cm_normalized = calc_conf_matrix(depvar, pred_vs_obs, limit_value, tweak_value, plot=False)
    tn, fp, fn, tp = cm_normalized.ravel()
    f1 = 2 * tp / (2 * tp + fp + fn)

    return tn, fp, fn, tp, f1
