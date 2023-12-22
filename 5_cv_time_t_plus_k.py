"""
This script performs temporal cross validation for model evaluation.
For a selected number of folds a model is trained with the parameters to be evaluated.
The predictions for each fold can be written to the DB.
The metrics per fold can be logged, however the actual evaluation is calculated on the
predictions written to the DB and can be seen in a Redash Dashboard.
"""

import logging
import time
from logging.config import dictConfig

import pandas as pd

from fairqmodel.build_splits import prepare_folded_input
from fairqmodel.command_line_args import get_command_args
from fairqmodel.create_model_description import create_model_description
from fairqmodel.data_preprocessing import (
    cap_high_values,
    cap_outliers,
    drop_stations_without_this_depvar,
    fix_column_types,
)
from fairqmodel.db_connect import send_data_clickhouse
from fairqmodel.feature_selection import assign_features_to_stage
from fairqmodel.model_parameters import get_lags, get_train_date_min, get_variables
from fairqmodel.model_wrapper import ModelWrapper
from fairqmodel.prediction_lag_adjusted import make_lag_adjusted_prediction
from fairqmodel.read_write_model_db import save_model_to_db
from fairqmodel.retrieve_data import retrieve_data
from fairqmodel.time_features import time_features
from fairqmodel.time_handling import get_current_local_time, to_unix
from logging_config.logger_config import get_logger_config

dictConfig(get_logger_config())

script_start_time = time.time()

# Set write_to_db to True to write results to clickhouse
write_to_db = get_command_args("write_db") or False
calc_metrics = True
use_two_stages = True

# Parameters for cv
n_cv_windows = 365
n_pred_days = 4
test_cv_window_size = int(24 * n_pred_days)
step_size = 24  # how much each prediction window is shifted against the previous one
train_shift = 30  # number of windows that use the same model
prediction_hour = 5  # in Berlin Time


# Select dependent variable
depvar = get_command_args("depvar") or "no2"  # no2, pm10, pm25

# Toggle development mode
dev = False

# Choose lags
use_lags = get_command_args("use_lags") or False

lags_actual, lags_avg = get_lags(use_lags=use_lags, selected_lags=[24, 48], lags_avg=[1, 2, 3, 4, 5])

# Retrieve and pre-process data from the DB
date_min = pd.Timestamp(get_train_date_min(depvar), tz="Europe/Berlin")
date_max = pd.Timestamp("2023-01-31", tz="Europe/Berlin")
n_train_years = int((date_max - date_min).total_seconds() / (3600 * 24 * 365))

logging.info("Started '5_cv_time_t_plus_k' for {}".format(depvar))
logging.info(
    "CV is performed for {} folds each containing {} days with a steps size of {}h".format(
        n_cv_windows, n_pred_days, step_size
    )
)

dat = retrieve_data(
    mode="stations",
    date_time_min=date_min,
    date_time_max=date_max,
    only_active_stations=False,
)
dat = drop_stations_without_this_depvar(dat, depvar)

# Remove outliers
dat = cap_outliers(dat)

# Create time features for each data point
dat = time_features(dat, depvar=depvar, lags_actual=lags_actual, lags_avg=lags_avg)

# Remove None value rows for the selected depvar.
non_missing_rows = dat.loc[:, depvar].notna()
dat = dat.loc[non_missing_rows, :].reset_index(drop=True)

# Select variables and fix dtypes
feature_cols, metric_feature_cols, categorical_feature_cols = get_variables(depvar, lags_actual, lags_avg, dev=dev)

dat = fix_column_types(dat, categorical_feature_cols, metric_feature_cols)


# Assign features to stages
features_stage_1, features_stage_2 = assign_features_to_stage(use_two_stages, feature_cols)

# TimeSeries CV for t+k
time_cv_folds = prepare_folded_input(
    dat,
    n_cv_windows=n_cv_windows,
    n_train_years=n_train_years,
    test_cv_window_size=test_cv_window_size,
    prediction_hour=prediction_hour,
    step_size=step_size,
)

# Store start time of training for DB entry
date_time_training_execution = get_current_local_time()
logging.info(f"date_time_training_execution: '{date_time_training_execution}'")

# This loop evaluates the quality of temporal predictions
loop_idx = 0
for fold in reversed(time_cv_folds):
    # Train a model only every x iterations
    if (loop_idx) % train_shift == 0:
        dat_train = fold["train"]

        models = ModelWrapper(
            depvar=depvar,
            feature_cols_1=features_stage_1,
            feature_cols_2=features_stage_2,
        )
        models.train(dat=cap_high_values(dat_train.copy(deep=True), depvar))

        max_train_date = fold["ts_fold_max_train_date"].strftime("%Y-%m-%d %H:%M:%S")
    # Log the progress
    ts_fold_id = fold["ts_fold_id"]
    max_test_date = fold["ts_fold_max_test_date"].strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"{ts_fold_id=}\t{max_train_date=}\t{max_test_date=}")

    # Eval the current fold
    all_results, _ = make_lag_adjusted_prediction(
        fold, models, feature_cols, depvar, lags_actual, lags_avg, calc_metrics=calc_metrics
    )
    if write_to_db:
        # Write models to DB
        if (loop_idx) % train_shift == 0:
            model_name = f"temporal_cv_t+{test_cv_window_size}"
            model_1_description, model_2_description = create_model_description(
                models, dat_train, lags_actual, lags_avg
            )

            model_id = save_model_to_db(
                models=models,
                model_name=model_name,
                model_1_description=model_1_description,
                model_2_description=model_2_description,
                execution_time=date_time_training_execution,
            )

        # Write prediction to DB
        df_predictions = all_results.rename(columns={"pred": "value"}).drop(columns=[depvar])
        df_predictions["model_id"] = model_id

        # Convert date columns to unix format
        df_predictions.loc[:, "date_time"] = to_unix(df_predictions["date_time"])
        df_predictions.loc[:, "date_time_forecast"] = to_unix(df_predictions["date_time_forecast"])

        # Reorder columns
        df_predictions = df_predictions[["model_id", "date_time_forecast", "date_time", "station_id", "value"]]

        send_data_clickhouse(df=df_predictions, table_name="model_predictions_temporal_cv", mode="replace")
    loop_idx += 1


script_end_time = time.time()
total_time = (script_end_time - script_start_time) // 60
logging.info("Finished script in total time of ~ {total_time} minutes.")
