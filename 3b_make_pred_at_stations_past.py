"""
This script performs temporal predictions at the stations for a selected time window.
The time window can, but doesn't have to, lie in the past.
A pre-trained model is loaded from the DB.
The predictions can be written to the DB.
This script is e.g. used for model or traffic-adjustment evaluation, for which the ground truth is known
and hence metrics can be calculated. The created predictions are necessary to calculate the tweak values for the
traffic thresholds.
Potentially this script could also be used for other tasks that require past predictions.
"""

import logging
from logging.config import dictConfig

import pandas as pd

from fairqmodel.command_line_args import get_command_args
from fairqmodel.data_preprocessing import cap_outliers, drop_stations_without_this_depvar, fix_column_types
from fairqmodel.db_connect import db_connect_target, get_query
from fairqmodel.prediction_t_plus_k import get_model_settings, prediction_t_plus_k
from fairqmodel.read_write_model_aux_functions import model_name_str
from fairqmodel.retrieve_data import retrieve_data
from fairqmodel.time_features import time_features
from fairqmodel.time_handling import get_current_local_time
from logging_config.logger_config import get_logger_config

dictConfig(get_logger_config())

# Configuration
DEV = False
verbose = True

# passed command args sample: ['filename.py', 'depvar=no2', 'write_db=True']
depvar = get_command_args("depvar") or "no2"
write_db = get_command_args("write_db") or False


if depvar == "no2":
    date_min = "2015-01-01 01:00:00"
elif depvar == "pm10":
    date_min = "2016-02-29 01:00:00"
else:  # pm25
    date_min = "2017-01-01 01:00:00"

date_max = str(get_current_local_time().strftime("%Y-%m-%d"))

# Retrieve and preprocess data
dat = retrieve_data(
    mode="stations",
    date_time_max=pd.Timestamp(date_max, tz="Europe/Berlin"),
    date_time_min=pd.Timestamp(date_min, tz="Europe/Berlin"),
)
n_days = int(
    (pd.Timestamp(date_max, tz="Europe/Berlin") - pd.Timestamp(date_min, tz="Europe/Berlin")) / pd.Timedelta(days=1)
)

# Remove outliers
dat = cap_outliers(dat)

model_type = "temporal"  # "temporal", "spatial", "all"

# Retrieve selected model and settings
with db_connect_target() as db:
    model_id = db.query_dataframe(get_query("final_model_id"), params={"model_type": model_type, "depvar": depvar})

model_id = model_id.model_id[0]

query_params = {"model_type": model_name_str(model_type), "depvar": depvar}

with db_connect_target() as db:
    available_models = db.query_dataframe(get_query("available_models"), params=query_params)

model_settings = get_model_settings(available_models, model_id)


logging.info(
    "Started '3b_make_pred_at_stations_past' for {} with model_id {} for time frame: [{}, {}]".format(
        depvar, model_id, date_min, date_max
    )
)


models = model_settings["models"]
depvar = model_settings["depvar"]
lags_actual = model_settings["lags"]
lags_avg = model_settings["lags_avg"]
feature_cols = model_settings["feature_cols"]
categorical_feature_cols = model_settings["categorical_feature_cols"]
metric_feature_cols = model_settings["metric_feature_cols"]

dat = drop_stations_without_this_depvar(dat, depvar)
dat = time_features(dat, depvar=depvar, lags_actual=lags_actual, lags_avg=lags_avg)

# Remove None value rows for the selected depvar.
non_missing_rows = dat.loc[:, depvar].notna()
dat = dat.loc[non_missing_rows, :].reset_index(drop=True)

dat = fix_column_types(dat, categorical_feature_cols, metric_feature_cols)

t_plus_k_params = {
    "n_windows": n_days,  # 365 for 1 year predictions  # perform <n_windows> forecasts
    "n_train_years": 0,  # model is already trained, hence no further training
    "window_size": 24,  # for each of the <n_windows> forecasts, perform prediction for upcoming <window_size> hours
    "step_size": 24,  # two adjacent forecasts are <step_size> hours apart
    "prediction_hour": 15,  # the prediction is performed at the time of execution.
}

table_name = "model_predictions_stations_lags"

prediction_t_plus_k(
    dat,
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
    calc_metrics=verbose,
)

logging.info(f"Finished predictions with depvar: {depvar}")
