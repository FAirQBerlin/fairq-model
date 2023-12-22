"""
This script trains a model with given hyper parameters and afterwards saves it to the DB.
No optimization or evaluation (CV) is performed.
"""

import logging
from logging.config import dictConfig

import pandas as pd

from fairqmodel.command_line_args import get_command_args
from fairqmodel.create_model_description import create_model_description
from fairqmodel.data_preprocessing import cap_high_values, cap_outliers, fix_column_types
from fairqmodel.feature_selection import assign_features_to_stage
from fairqmodel.model_parameters import get_lags, get_train_date_min, get_variables
from fairqmodel.model_wrapper import ModelWrapper
from fairqmodel.read_write_model_db import save_model_to_db, write_model_to_models_final
from fairqmodel.retrieve_data import retrieve_data
from fairqmodel.time_features import time_features
from fairqmodel.time_handling import get_current_local_time
from logging_config.logger_config import get_logger_config

dictConfig(get_logger_config())

# passed command args sample: ['filename.py', 'depvar=no2', 'use_lags=True','update_models_final=True']
depvar = get_command_args("depvar") or "no2"
update_models_final = get_command_args("update_models_final") or False
use_lags = get_command_args("use_lags") or False
# Caution: default True cannot be set that way because it may become "False or True" which is True

use_two_stages = True  # True for models used for traffic adjustment simulation

# Toggle development mode
dev = False

# Retrieve and pre-process data
date_min = get_train_date_min(depvar)

date_max = str(get_current_local_time().strftime("%Y-%m-%d"))

logging.info(
    "Started '2_train' for {} with: \n - use_two_stages: {}, \n - use_lags: {}, \n - time frame: [{}, {}]".format(
        depvar, use_two_stages, use_lags, date_min, date_max
    )
)

# Select lag Information
lags_actual, lags_avg = get_lags(use_lags=use_lags)

# --- End of parameter selection

# Get variables
feature_cols, metric_feature_cols, categorical_feature_cols = get_variables(depvar, lags_actual, lags_avg, dev=dev)

# Assign features to model stages
features_stage_1, features_stage_2 = assign_features_to_stage(use_two_stages, feature_cols)

# Retrieve the data
dat_original = retrieve_data(
    mode="stations",
    date_time_max=pd.Timestamp(date_max, tz="Europe/Berlin"),
    date_time_min=pd.Timestamp(date_min, tz="Europe/Berlin"),
    only_active_stations=False,
)

# Remove outliers
dat_original = cap_outliers(dat_original)

# Remove missing entries
non_missing_rows = dat_original.loc[:, depvar].notna()
dat_original = dat_original.loc[non_missing_rows, :].reset_index(drop=True)

# Create time features for each data point
dat = time_features(dat_original.copy(deep=True), depvar=depvar, lags_actual=lags_actual, lags_avg=lags_avg)

dat = fix_column_types(dat, categorical_feature_cols, metric_feature_cols)

# Train model
date_time_training_execution = get_current_local_time()
models = ModelWrapper(
    depvar=depvar,
    feature_cols_1=features_stage_1,
    feature_cols_2=features_stage_2,
)
models.train(dat=cap_high_values(dat.copy(deep=True), depvar))

domain = "temporal" if use_lags else "spatial"
model_name = f"full_data_{domain}"

model_1_description, model_2_description = create_model_description(models, dat, lags_actual, lags_avg)

model_id = save_model_to_db(
    models=models,
    model_name=model_name,
    model_1_description=model_1_description,
    model_2_description=model_2_description,
    execution_time=date_time_training_execution,
)
logging.info(f"Training of model with id {model_id} completed")

if update_models_final:
    # If you want to use this model for the daily predictions from now on:
    write_model_to_models_final(depvar, domain, model_id)
