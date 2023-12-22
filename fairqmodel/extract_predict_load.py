import logging
from decimal import Decimal
from logging.config import dictConfig
from typing import Optional

import numpy as np
import pandas as pd

from fairqmodel.data_preprocessing import cap_outliers, fix_column_types
from fairqmodel.db_connect import send_data_clickhouse
from fairqmodel.prediction_kfz_adjusted import adjust_kfz_per_hour_grid
from fairqmodel.retrieve_data import check_number_of_rows, retrieve_data
from fairqmodel.time_handling import to_unix
from logging_config.logger_config import get_logger_config

dictConfig(get_logger_config())


def extract_predict_load_grid(
    batch: int,
    date_time_forecast: pd.Timestamp,
    date_time_max: pd.Timestamp,
    model_settings: dict,
    model_id: Optional[int] = None,
    write_db: Optional[bool] = False,
    mode: Optional[str] = "grid",
) -> dict:
    """
    Extracts, predicts and loads the data for one batch (grid model).

    :param batch: int, Batch number, pointing to a set of coordinates
    :param date_time_forecast: Optional[pd.Timestamp], timestamp the forecast ist made,
    equals the minimal timestamp to select
    :param date_time_max: Optional[pd.Timestamp], Maximal timestamp to select
    :param model_settings: dict, dictionary of model settings, see get_model_settings(),
    at least the fields models, categorical_feature_cols and metric_feature_cols must be provided
    :param model_id: Optional[int], id of the model, us used and has to be set if the model
    is written to db. None per default.
    :param write_db: Optional[bool], write the predictions of this model_id to db?
     per default False. target table: table model_predictions_grid
    :param mode: Optional[str] = "grid". Either "grid" for normal mode, or "grid_sim" for simulation of kfz values.

    return: dictionary of batch and finished (e.g. {'batch': 2, 'finished': True}),
    indicating whether the respective batch was finished successfully
    """
    if mode not in ["grid", "grid_sim"]:
        raise ValueError(f"Mode must be one of: grid, grid_sim, but is mode = {mode}")

    logging.info(f"Retrieving data for batch {batch} in mode {mode}")

    models = model_settings["models"]
    categorical_feature_cols = model_settings["categorical_feature_cols"]
    metric_feature_cols = model_settings["metric_feature_cols"]

    dat = retrieve_data(
        mode=mode,
        batch=batch,
        date_time_min=date_time_forecast,
        date_time_max=date_time_max,
    )

    rows_ok = check_number_of_rows(batch, mode, dat, date_time_forecast, date_time_max)

    dat = cap_outliers(dat)

    dat_features = fix_column_types(dat, categorical_feature_cols, metric_feature_cols)

    logging.info(f"Making predictions for batch {batch}")

    if mode == "grid_sim":
        percentages = np.arange(0, 101, 10).tolist()
    else:
        percentages = [100]

    table_name = f"model_predictions_{mode}"
    send_ok = []

    for percentage in percentages:  # Loop will be of length one if we're not making the kfz simulation
        dat_features_pct = adjust_kfz_per_hour_grid(dat_features, percentage)

        predictions, _, _ = models.predict(dat=dat_features_pct)

        if write_db:
            assert model_id is not None
            df_for_db = prepare_preds_for_db(dat_features, predictions, model_id, date_time_forecast, mode, percentage)
            this_send_ok = write_preds_to_db(df_for_db, table_name)
            send_ok.append(this_send_ok)
        else:
            send_ok.append(True)

        logging.info(f"Wrote predictions to DB for batch {batch}")

    logging.info(f"Finished batch {batch}\n")

    batch_report = {"batch": batch, "finished": rows_ok & all(send_ok)}

    return batch_report


def prepare_preds_for_db(
    dat_features: pd.DataFrame,
    predictions: np.ndarray,
    model_id: int,
    date_time_forecast: pd.Timestamp,
    mode: Optional[str] = "grid",
    percentage: Optional[int] = 100,
) -> pd.DataFrame:
    """
    Prepare Dataframe of predictions for db

    :param dat_features: pd.DataFrame with the features
    :param predictions: array of the predictions from model.predict, same order as dat_features
    :param model_id: model_id
    :param date_time_forecast: date time the forecast was made
    :param mode: Optional[str] = grid. Either "grid" (default) for normal mode, or "grid_sim" for kfz_simulation.
    :param percentage: Optional[int] = 100. Percentage to adjust the kfz_per_hour. 100% per default (no adjustment)


    return: bool True if succeeded, false otherwise
    """
    df_for_db = dat_features.loc[:, ["date_time", "x", "y"]]
    df_for_db["model_id"] = model_id
    df_for_db["date_time_forecast"] = date_time_forecast
    df_for_db["value"] = predictions
    df_for_db["value"] = round(df_for_db["value"], 1).apply(Decimal)
    if mode == "grid_sim":
        df_for_db["kfz_pct"] = percentage
        df_for_db = df_for_db.loc[:, ["model_id", "date_time_forecast", "date_time", "x", "y", "kfz_pct", "value"]]
    elif mode == "grid":
        df_for_db = df_for_db.loc[:, ["model_id", "date_time_forecast", "date_time", "x", "y", "value"]]
    else:
        raise ValueError(f"Mode must be one of: grid, grid_sim, but is mode = {mode}")

    return df_for_db


def write_preds_to_db(df_for_db: pd.DataFrame, table_name: str) -> bool:
    """
    Write the predictions to db

    :param df_for_db: pd.DataFrame DataFrame to send to DB
    :param table_name: str name of the table where to insert the data

    return: bool True if succeeded, false otherwise
    """
    logging.info(f"Preparing to send to db first_date_time = {min(df_for_db.loc[:,'date_time'])}")
    logging.info(f"Preparing to send to db date_time_forecast = {df_for_db.loc[0,'date_time_forecast']}")
    logging.info(f"Preparing to send to db last_date_time = {max(df_for_db.loc[:,'date_time'])}")

    # Convert date columns to unix format
    df_for_db.loc[:, "date_time"] = to_unix(df_for_db["date_time"])
    df_for_db.loc[:, "date_time_forecast"] = to_unix(df_for_db["date_time_forecast"])

    logging.info("Writing predictions to DB")

    # Send results to the DB
    send_ok = send_data_clickhouse(df=df_for_db, table_name=table_name, mode="insert")

    return send_ok


def get_batches_to_reschedule(reporting: list[dict]):
    """
    Get batches that have never been successful in the reporting list.
    Batches that have been successful once, and once (or several times) not, do not get rescheduled

    :param reporting: list(dict('batch': batch, 'finished': True)) reporting list of dictionaries,
    each dict is returned by extract_predict_load and indicates the 'batch' and if it 'finished' successfully.

    return: list of batch ids to reschedule
    """

    successful_batches = [batch["batch"] for batch in reporting if batch["finished"] is True]
    reschedule = [batch["batch"] for batch in reporting if batch["batch"] not in successful_batches]

    return reschedule
