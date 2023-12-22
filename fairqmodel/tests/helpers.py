import os
import pickle
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd

from fairqmodel.data_preprocessing import fix_column_types
from fairqmodel.time_features import time_features


def arrange_for_lag_pred_tests() -> Tuple[pd.DataFrame, dict, pd.Timestamp, List[str]]:
    """Performs the arrange step from the test_pre_fill_lags function.
    A pre-trained model is loaded and a dummy DataFrame is constructed.
    The returned variables are:
        - pd.DataFrame, Containing dummy data
        - Dict, Containing the pre-trained model and all its meta-information
        - pd.Timestamp, An arbitrarily chosen 'min_date'
        - List[str], Containing the names of columns that should be modified by the tested function

    :return: Tuple[pd.DataFrame, dict, pd.Timestamp, List[str]]
    """
    # Load a pre-trained dummy model and retrieve its parameters
    script_dir = os.path.dirname(__file__)
    rel_path = "models_and_data/test_model_settings.pickle"
    abs_file_path = os.path.join(script_dir, rel_path)

    with open(abs_file_path, "rb") as handle:
        model_settings = pickle.load(handle)

    depvar = model_settings["depvar"]
    features = list(set(model_settings["feature_cols"]))
    categorical_feature_cols = list(set(model_settings["categorical_feature_cols"]))
    metric_feature_cols = list(set(model_settings["metric_feature_cols"]))
    lags_actual = model_settings["lags"]
    lags_avg = model_settings["lags_avg"]

    # Number of time points in the DataFrame
    time_points = 25

    # Number of time points for which 'observations' are available
    observed_time_points = 10

    # Values for all variables are set to one
    data = np.ones((time_points, len(features)))

    # Create the DataFrame
    dat = pd.DataFrame(data, columns=features)

    # Fill the devpar column with available 'observations'
    dat.loc[:, depvar] = np.arange(0, observed_time_points).tolist() + [None] * (time_points - observed_time_points)

    # Set an arbitrary station_id to enable the lag construction
    dat.loc[:, "station_id"] = "314"

    # Fill the 'date_time' column with real but arbitrary dates
    dat.loc[:, "date_time"] = np.arange(datetime(2021, 1, 1), datetime(2022, 1, 1), timedelta(hours=1)).astype(
        datetime
    )[:time_points]

    # Select the date that is used as 'min_date' for the pre_fill function
    date_min = pd.Timestamp("2021-01-01 03:00:00")

    # Preprocess the data
    dat = time_features(dat, depvar=depvar, lags_actual=lags_actual, lags_avg=lags_avg)
    dat = fix_column_types(dat, categorical_feature_cols, metric_feature_cols)

    changed_columns = ["pm25_lag1", "pm25_lag2", "pm25_lag3", "lag_avg_(1, 2)"]

    return dat, model_settings, date_min, changed_columns
