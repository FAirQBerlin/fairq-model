"""Extract data, predict, and load results to the database for grid batches."""

from decimal import Decimal

import numpy as np
import pandas as pd
from loguru import logger

from fairqmodel.data_preprocessing import cap_outliers, fix_column_types
from fairqmodel.db_connect import send_data_clickhouse
from fairqmodel.prediction_kfz_adjusted import adjust_kfz_per_hour_grid
from fairqmodel.retrieve_data import check_number_of_rows, retrieve_data


def extract_predict_load_grid(
    batch: int,
    date_time_forecast: pd.Timestamp,
    date_time_max: pd.Timestamp,
    model_settings: dict,
    model_id: int | None = None,
    write_db: bool | None = False,
    mode: str | None = "grid",
) -> dict:
    """Extract, predict, and load the data for one batch (grid model).

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

    logger.info(f"Retrieving data for batch {batch} in mode {mode}")

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

    logger.info(f"Making predictions for batch {batch}")

    percentages = np.arange(0, 101, 10).tolist() if mode == "grid_sim" else [100]

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

        logger.info(f"Wrote predictions to DB for batch {batch}")

    logger.info(f"Finished batch {batch}\n")

    return {"batch": batch, "finished": rows_ok & all(send_ok)}


def prepare_preds_for_db(
    dat_features: pd.DataFrame,
    predictions: np.ndarray,
    model_id: int,
    date_time_forecast: pd.Timestamp,
    mode: str | None = "grid",
    percentage: int | None = 100,
) -> pd.DataFrame:
    """Prepare a DataFrame of predictions for the database.

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
    """Write the predictions to the database.

    :param df_for_db: pd.DataFrame DataFrame to send to DB
    :param table_name: str name of the table where to insert the data

    return: bool True if succeeded, false otherwise
    """
    logger.info(f"Preparing to send to db first_date_time = {min(df_for_db.loc[:, 'date_time'])}")
    logger.info(f"Preparing to send to db date_time_forecast = {df_for_db.loc[0, 'date_time_forecast']}")
    logger.info(f"Preparing to send to db last_date_time = {max(df_for_db.loc[:, 'date_time'])}")

    logger.info("Writing predictions to DB")

    # Send results to the DB
    return send_data_clickhouse(df=df_for_db, table_name=table_name, mode="insert")
