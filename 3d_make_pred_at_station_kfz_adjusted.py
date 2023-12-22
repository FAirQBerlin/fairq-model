"""
This script performs temporal predictions at the stations for a selected time window.
The time window can, but doesn't have to, lay in the past.
Different to script 3b, the 'kfz_per_hour' can be modified for selected hours of the day.
A pre-trained model is loaded from the DB.
The predictions can be written to the DB.
This script is e.g. used to emulate or suggest kfz-reductions such that limit values for the pollutant are not exceeded.
"""

import logging
from logging.config import dictConfig

from fairqmodel.command_line_args import get_command_args
from fairqmodel.db_connect import db_connect_target, get_query
from fairqmodel.prediction_kfz_adjusted import prediction_kfz_adjusted
from fairqmodel.prediction_t_plus_k import get_model_settings
from fairqmodel.read_write_model_aux_functions import model_name_str
from fairqmodel.time_handling import get_current_local_time
from logging_config.logger_config import get_logger_config

dictConfig(get_logger_config())

# Select general settings
depvar = get_command_args("depvar") or "no2"
write_db = get_command_args("write_db") or False
forecast_days = get_command_args("forecast_days") or 4

logging.info("Starting with depvar = {}, write_db = {}, forecast_days = {}".format(depvar, write_db, forecast_days))

model_type = "temporal"  # "temporal", "spatial", "all"
two_stages = True  # Only two-staged models are working well for the kfz-adjustment

# Retrieve selected model and settings
query_params_model_id = {"model_type": model_type, "depvar": depvar}
with db_connect_target() as db:
    model_id = db.query_dataframe(get_query("final_model_id"), params=query_params_model_id)

model_id = model_id.model_id[0]

query_params_model = {"model_type": model_name_str(model_type), "depvar": depvar}
with db_connect_target() as db:
    available_models = db.query_dataframe(get_query("available_models"), params=query_params_model)

model_settings = get_model_settings(available_models, model_id)

# Select Settings for prediction
date_min = get_current_local_time().strftime("%Y-%m-%d")

# If None, predictions for [0, 10, ..., 100] percent are made, else only for the given value
kfz_percentage = None

# All currently relevant station ids
station_ids = ["117", "124", "143", "174"]

for station_id in station_ids:
    logging.info(
        "Started kfz-adjusted predictions for {} with model_id {} from {} for {} days at station {}".format(
            depvar, model_id, date_min, forecast_days, station_id
        )
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
    logging.info(f"Finished predictions for station {station_id}\n")

logging.info(f"Finished predictions with depvar: {depvar}")
