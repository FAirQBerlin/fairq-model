"""
In this script, the model is repeatedly fitted.
In each step, it is fitted on all but one stations. Then we make predictions for that one station.
This helps to estimate the model performance when making predictions for grid cells we don't have measures for.
The predictions are only for t itself (not into the future) because we focus on the spatial aspect here.
The model does not include lags because there are no lags available away the stations.
The predictions are written to the database to be used in a Dashboard.
"""

import logging
from logging.config import dictConfig

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fairqmodel.command_line_args import get_command_args
from fairqmodel.create_model_description import create_model_description
from fairqmodel.data_preprocessing import (
    cap_high_values,
    cap_outliers,
    drop_stations_without_this_depvar,
    fix_column_types,
)
from fairqmodel.db_connect import db_connect_source, get_query, send_data_clickhouse
from fairqmodel.feature_selection import assign_features_to_stage
from fairqmodel.model_parameters import get_lags, get_train_date_min, get_variables
from fairqmodel.model_wrapper import ModelWrapper
from fairqmodel.read_write_model_db import save_model_to_db
from fairqmodel.retrieve_data import retrieve_data
from fairqmodel.time_features import time_features
from fairqmodel.time_handling import get_current_local_time, to_unix
from logging_config.logger_config import get_logger_config

dictConfig(get_logger_config())

use_two_stages = True

# Select dependent variable
depvar = get_command_args("depvar") or "no2"  # no2, pm10, pm25
write_db = get_command_args("write_db") or False


# Get data and station_types from DB
with db_connect_source() as db:
    id_type_mapping_df = db.query_dataframe(get_query("id_type_mapping"))

id_type_mapping = dict(zip(id_type_mapping_df.id, id_type_mapping_df.stattyp))

# Retrieve and pre-process data from the DB
date_min = get_train_date_min(depvar)


date_max = "2023-11-27 01:00:00"

logging.info("Started '4_cv_loso' for {} for time frame: [{}, {}]".format(depvar, date_min, date_max))

dat = retrieve_data(
    mode="stations",
    date_time_max=pd.Timestamp(date_max, tz="Europe/Berlin"),
    date_time_min=pd.Timestamp(date_min, tz="Europe/Berlin"),
    only_active_stations=False,
)
dat = drop_stations_without_this_depvar(dat, depvar)

# Remove outliers
dat = cap_outliers(dat)

# Toggle development mode
dev = False

lags_actual, lags_avg = get_lags(use_lags=False)

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

# Store start time of training for DB entry
date_time_training_execution = get_current_local_time()

all_station_ids = dat.station_id.unique().tolist()


# Loop compares how good left-out stations are estimated.
for station_id in all_station_ids:
    logging.info(f"Now predicting station '{station_id}'")
    station_type = id_type_mapping[station_id]

    # Split data into train and test splits
    dat_train = dat[dat["station_id"] != station_id]
    dat_test = dat[dat["station_id"] == station_id]

    # Initialize the model
    models = ModelWrapper(
        depvar=depvar,
        feature_cols_1=features_stage_1,
        feature_cols_2=features_stage_2,
    )

    # Train the model
    models.train(dat=cap_high_values(dat_train.copy(deep=True), depvar))

    # Perform predictions and calculate metrics
    predictions, pred_stage_1, pred_stage_2 = models.predict(dat_test)
    label = dat_test[depvar].tolist()

    rmse = mean_squared_error(label, predictions, squared=False)
    mae = mean_absolute_error(label, predictions)
    r2 = r2_score(label, predictions)

    logging.info(f"{station_id=}\t{rmse=:.3f}\t{mae=:.3f}\t{r2=:.3f}\t{station_type=}")

    if write_db:
        # Write models to DB
        model_name = f"spatial_cv_without_station_{station_id}"

        model_1_description, model_2_description = create_model_description(models, dat_train, lags_actual, lags_avg)
        model_id = save_model_to_db(
            models=models,
            model_name=model_name,
            model_1_description=model_1_description,
            model_2_description=model_2_description,
            execution_time=date_time_training_execution,
        )

        # Write predictions to DB
        df_predictions = dat_test.loc[:, ["station_id", "date_time"]].copy(deep=True)
        df_predictions["value"] = predictions
        df_predictions["model_id"] = model_id

        # Convert date column to unix format
        df_predictions.loc[:, "date_time"] = to_unix(df_predictions["date_time"])

        # Reorder columns
        df_predictions = df_predictions[["model_id", "date_time", "station_id", "value"]]

        send_data_clickhouse(df=df_predictions, table_name="model_predictions_spatial_cv", mode="replace")
