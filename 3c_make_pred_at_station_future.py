"""
This script performs temporal predictions at the stations for the future.
The prediction is performed from the execution date, up to a selected number of days into the future.
A pre-trained model is loaded from the DB.
The predictions can be written to the DB.

NOTE: Script is used for daily jobs -> Don't change the model_id.
"""

import logging
from datetime import timezone
from logging.config import dictConfig

import numpy as np
import pandas as pd

from fairqmodel.command_line_args import get_command_args
from fairqmodel.data_preprocessing import cap_outliers, drop_stations_without_this_depvar, fix_column_types
from fairqmodel.db_connect import db_connect_target, get_query
from fairqmodel.get_max_date_time import get_max_date_time
from fairqmodel.prediction_t_plus_k import get_model_settings, prediction_t_plus_k
from fairqmodel.read_write_model_aux_functions import model_name_str
from fairqmodel.retrieve_data import retrieve_data
from fairqmodel.time_features import time_features
from fairqmodel.time_handling import get_model_start_time, timestamp_to_np_datetime64
from logging_config.logger_config import get_logger_config

dictConfig(get_logger_config())

# Configuration
DEV = False
verbose = True

model_type = "temporal"  # "temporal", "spatial"

# passed command args sample: ['filename.py', 'depvar=no2', 'write_db=True']
depvar = get_command_args("depvar") or "no2"
write_db = get_command_args("write_db") or False

logging.info("Starting with depvar = {}, write_db = {}".format(depvar, write_db))

max_lag_hours = 48

date_now = get_model_start_time(twice_daily=False)

# Select min and max date, s.t.
# - enough observation data is provided to build the lags,
# - enough prediction data is provided to allow for the forecast.
date_min = date_now - pd.Timedelta(max_lag_hours, "hour")
date_max = get_max_date_time(date_now)

# Get data from the DB
dat = retrieve_data(mode="stations", date_time_max=date_max, date_time_min=date_min)
dat = drop_stations_without_this_depvar(dat, depvar)

# Remove outliers
dat = cap_outliers(dat)

# If a larger number of hours for the forecast is selected than available,
# select all available data.
# NOTE: dat.date_time column is in UTC, but time zone naive -> Specify before calculation
forecast_hours = -(date_now - dat.date_time.max().tz_localize(timezone.utc)) / np.timedelta64(1, "h")

# Retrieve selected model and settings
with db_connect_target() as db:
    model_id = db.query_dataframe(get_query("final_model_id"), params={"model_type": model_type, "depvar": depvar})

model_id = model_id.model_id[0]

logging.info("Using model with id {}, predicting the upcoming {} hours".format(model_id, int(forecast_hours)))

query_params = {"model_type": model_name_str(model_type), "depvar": depvar}

with db_connect_target() as db:
    available_models = db.query_dataframe(get_query("available_models"), params=query_params)

model_settings = get_model_settings(available_models, model_id)

models = model_settings["models"]
depvar = model_settings["depvar"]
lags_actual = model_settings["lags"]
lags_avg = model_settings["lags_avg"]
feature_cols = model_settings["feature_cols"]
categorical_feature_cols = model_settings["categorical_feature_cols"]
metric_feature_cols = model_settings["metric_feature_cols"]

t_plus_k_params = {
    "n_windows": 1,  # perform one prediction
    "n_train_years": 0,  # model is already trained, hence no further training
    "window_size": forecast_hours,  # prediction selected number of hours
    "step_size": 0,  # since there is just one prediction, the step_size is not relevant here
    "prediction_hour": None,  # the prediction is performed at the time of execution.
}

# Specify the DB target table
table_name = "model_predictions_temporal"

all_stations = list(dat.station_id.unique())
logging.info(f"Predictions will be performed for the stations: \n \t {all_stations}")

# Perform the forecast for one station at a time.
for station in all_stations:
    logging.info(f"Current 'station_id': {station}")
    dat_station = dat.query(f"station_id == '{station}'").copy(deep=True)
    dat_station = time_features(dat_station, depvar=depvar, lags_actual=lags_actual, lags_avg=lags_avg)
    # Convert 'date_now' to UTC before it is used to query the DataFrame
    time_without_tz = timestamp_to_np_datetime64(date_now)
    dat_station.query(f"date_time >='{time_without_tz}'", inplace=True)
    dat_station = fix_column_types(dat_station.copy(deep=True), categorical_feature_cols, metric_feature_cols)
    dat_station.reset_index(level=0, inplace=True, drop=True)
    prediction_t_plus_k(
        dat_station,
        models,
        model_id,
        feature_cols,
        depvar,
        lags_actual,
        lags_avg,
        t_plus_k_params,
        table_name,
        write_db,
        verbose,
        calc_metrics=False,  # can't be calculated for future data
        include_current_time_point=True,
    )
    logging.info(f"Finished 'station_id': {station}\n")

logging.info(f"Finished predictions with depvar: {depvar}")
