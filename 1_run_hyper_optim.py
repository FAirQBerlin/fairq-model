"""
This script optimizes the set of model hyper parameters.
Different combinations of parameters are considered using the Optuna tool.
Tested parameters and corresponding evaluation results are stored in a local DB.
To use them in future models, they must be updated in fairqmodel/params/xgboost_params.json].
"""

import pandas as pd

from fairqmodel.command_line_args import get_command_args
from fairqmodel.data_preprocessing import cap_outliers, drop_stations_without_this_depvar, fix_column_types
from fairqmodel.hyper_optim import hyper_opt
from fairqmodel.model_parameters import get_lag_options, get_train_date_min, get_variables
from fairqmodel.retrieve_data import retrieve_data
from fairqmodel.time_features import time_features
from fairqmodel.time_handling import get_current_local_time

# Select dependent variable
depvar = get_command_args("depvar") or "no2"  # no2, pm10, pm25

# Get data from the DB
date_min = get_train_date_min(depvar)

date_max = "2022-02-28 01:00:00"

dat = retrieve_data(
    mode="stations",
    date_time_max=pd.Timestamp(date_max, tz="Europe/Berlin"),
    date_time_min=pd.Timestamp(date_min, tz="Europe/Berlin"),
    only_active_stations=False,
)

dat = drop_stations_without_this_depvar(dat, depvar)

# Remove outliers
dat = cap_outliers(dat)

use_lags = False
optimize_lags = False

# Specifies for which stage of the model the parameters will be optimized.
optimize_stage = 1  # Allowed values: 1, 2

# If the parameters of the first stage are optimized, the model is treated as a one-stage model,
# but a different set of variables is used, hence 'use_two_stages' cannot be inferred from 'optimize_stage'
use_two_stages = True

# use_two_stages can't be False if optimize_stage == 2
assert use_two_stages or optimize_stage == 1

# If the parameters of the second stage are optimized, a pre-trained first stage must be selected.
model_id_dict = {"no2": 369, "pm25": 368, "pm10": 370}  # Optimized first stages with monotonic constraints
first_stage_model_id = model_id_dict[depvar] if use_two_stages else None


# Set parameters for the t+k evaluation
n_train_years = 5
n_cv_windows = 365  # On how many windows the parameters will be evaluated
n_pred_days = 4  # Size of each evaluation window
step_size = 24  # How much each evaluation window is shifted against the previous one
train_shift = 30  # Number of evaluation windows that predicted by the same model
prediction_hour = 14  # Hour of the day when the prediction is performed

# Toggle development mode.
dev = False

n_windows = n_cv_windows if not dev else 5  # to increase speed in development
k_hours = int(24 * n_pred_days)

t_plus_k_params = {
    "n_cv_windows": n_windows,
    "n_train_years": n_train_years,
    "test_cv_window_size": k_hours,
    "step_size": step_size,
    "prediction_hour": prediction_hour,
    "train_shift": train_shift,
}

num_boost_round = 500 if not dev else 50  # to increase speed in development
max_minutes = 60 * 24
n_cores = 8

lag_options, lags_avg = get_lag_options(use_lags, lags_avg=[1, 2, 3, 4, 5])


# Specifies the parameters to optimize and their value range.
# The structure is as follows: "hyper-parameter": [lower_bound, upper_bound].
# Currently, only the following hyper-parameters can be optimized:
params = {
    "eta": [0.02, 0.4],
    "max_depth": [5, 13],
    "gamma": [1e-7, 5],
    "min_child_weight": [50, 300],
    "subsample": [0.75, 0.95],  # sample 80% of the rows
    "colsample_bytree": [0.75, 0.95],  # sample 80% of the feature columns
}

if use_lags and optimize_lags:
    params["lag_idx"] = [0, len(lag_options) - 1]

# Choose lags
# Every possible lag column is created here.
# During optimization only a few of these are selected per trial.
lags_actual = [lag for lag in set([lag for option in lag_options for lag in option])]

# Create time features for each data point
dat = time_features(dat, depvar=depvar, lags_actual=lags_actual, lags_avg=lags_avg)

non_missing_rows = dat.loc[:, depvar].notna()
dat = dat.loc[non_missing_rows, :].reset_index(drop=True)

# Select variables and fix dtypes
feature_cols, metric_feature_cols, categorical_feature_cols = get_variables(depvar, lags_actual, lags_avg, dev=dev)
feature_cols_without_lags = [feature for feature in feature_cols if f"{depvar}_lag" not in feature]  # for study name

dat = fix_column_types(dat, categorical_feature_cols, metric_feature_cols)

study_name_dict = {
    "ExecutionDate": get_current_local_time().date(),
    "XDataShape": dat.shape,
    "DataDateRange": (date_min, date_max),
    "Prediction_hours": k_hours,
    "num_boost_round": num_boost_round,
    "max_minutes": max_minutes,
    "n_cores": n_cores,
    "depvar": depvar,
    "feature_cols": feature_cols_without_lags,
    "hpo_params": params,
    "use_lags": use_lags,
    "optimize_lags": optimize_lags,
    "optimize_stage": optimize_stage,
}

base_name = "XGBoost"

study_name = f"{base_name}-{study_name_dict}"

res_hyper_opt = hyper_opt(
    depvar=depvar,
    params=params,
    lag_options=lag_options,
    lags_avg=lags_avg,
    dat=dat,
    feature_cols=feature_cols,
    num_boost_round=num_boost_round,
    max_minutes=max_minutes,
    study_name=study_name,
    n_cores=n_cores,
    early_stopping_rounds=50,
    n_trials=None,
    silence=False,
    seed=123,
    t_plus_k_params=t_plus_k_params,
    storage=True,
    use_lags=use_lags,
    optimize_lags=optimize_lags,
    use_two_stages=use_two_stages,
    optimize_stage=optimize_stage,
    first_stage_model_id=first_stage_model_id,
)

# run via # pipenv run python 1_run_hyper_optim.py
# monitor via # pipenv run optuna-dashboard sqlite:///optuna_hpo_studies.db
# the optimal values found by the HPO are manually stored in /params/xgboost_params.json
