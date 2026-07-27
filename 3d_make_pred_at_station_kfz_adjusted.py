"""Perform temporal predictions at the stations for a selected time window with kfz adjustment.

The time window can, but doesn't have to, lay in the past.
Different to script 3b, the 'kfz_per_hour' can be modified for selected hours of the day.
A pre-trained model is loaded from the DB.
The predictions can be written to the DB.
This script is e.g. used to emulate or suggest kfz-reductions such that limit values for the pollutant are not exceeded.
"""

import sys

from loguru import logger

import fairqmodel as fqm
from fairqmodel.command_line_args import get_command_args
from fairqmodel.db_connect import db_connect_target, get_query
from fairqmodel.prediction_kfz_adjusted import prediction_kfz_adjusted
from fairqmodel.prediction_t_plus_k import get_model_settings
from fairqmodel.time_handling import get_current_local_time

logger.remove()
logger.add(sys.stdout, level="INFO")

logger.info(f"Using fairqmodel in version: {fqm.__version__}")

# Select general settings
depvar = get_command_args("depvar") or "no2"
write_db = get_command_args("write_db") or False
forecast_days = get_command_args("forecast_days") or 4

logger.info(f"Starting with depvar = {depvar}, write_db = {write_db}, forecast_days = {forecast_days}")

model_type = "temporal"  # "temporal", "spatial", "all"
two_stages = True  # Only two-staged models are working well for the kfz-adjustment

# Retrieve selected model and settings
query_params_model_id = {"model_type": model_type, "depvar": depvar}
with db_connect_target() as db:
    model_id = db.query_dataframe(get_query("final_model_id"), params=query_params_model_id)

model_id = model_id.model_id[0]

model_settings = get_model_settings(model_id)

# Select Settings for prediction
date_min = get_current_local_time().strftime("%Y-%m-%d")

# If None, predictions for [0, 10, ..., 100] percent are made, else only for the given value
kfz_percentage = None

# All currently relevant station ids
station_ids = ["117", "124", "174"]

for station_id in station_ids:
    logger.info(
        f"Started kfz-adjusted predictions for {depvar} with model_id {model_id}"
        f" from {date_min} for {forecast_days} days at station {station_id}"
    )
    # Perform prediction
    all_results, _ = prediction_kfz_adjusted(
        model_settings,
        date_min,
        station_id,
        model_id,
        forecast_days=forecast_days,
        kfz_percentage=kfz_percentage,
        write_to_db=write_db,
    )
    logger.info(f"Finished predictions for station {station_id}\n")

logger.info(f"Finished predictions with depvar: {depvar}")
