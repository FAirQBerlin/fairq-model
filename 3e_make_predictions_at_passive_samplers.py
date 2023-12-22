"""
Make predictions for all grid cells that contain a passive sampler for a given time frame.
These predictions are used to be compared with the values measured by the passive samplers
as well as the simulated values from the chemical transport model.
Caution: traffic predictions must be available in the database for the specified time period and grid cells.
"""
import pandas as pd

from fairqmodel.data_preprocessing import cap_outliers, fix_column_types
from fairqmodel.db_connect import db_connect_target, get_query, send_data_clickhouse
from fairqmodel.prediction_t_plus_k import get_model_settings
from fairqmodel.read_write_model_aux_functions import model_name_str
from fairqmodel.retrieve_data import retrieve_data
from fairqmodel.time_handling import get_model_start_time, to_unix

# Selectable parameters
DEV = False
write_db = False
depvar = "pm25"  # pm10, pm25, no2
date_min = "2022-01-01 00:00:00"
date_max = "2022-12-31 23:00:00"

# Retrieve model and settings
model_type = "spatial"
with db_connect_target() as db:
    model_id = db.query_dataframe(get_query("final_model_id"), params={"model_type": model_type, "depvar": depvar})
model_id = model_id.model_id[0]

query_params = {"model_type": model_name_str(model_type), "depvar": depvar}
with db_connect_target() as db:
    available_models = db.query_dataframe(get_query("available_models"), params=query_params)
model_settings = get_model_settings(available_models, model_id)

models = model_settings["models"]
depvar = model_settings["depvar"]
feature_cols = model_settings["feature_cols"]
categorical_feature_cols = model_settings["categorical_feature_cols"]
metric_feature_cols = model_settings["metric_feature_cols"]

# Get data
dat = retrieve_data(
    mode="passive_samplers",
    date_time_max=pd.Timestamp(date_max, tz="Europe/Berlin"),
    date_time_min=pd.Timestamp(date_min, tz="Europe/Berlin"),
)
dat = cap_outliers(dat)
dat_features = fix_column_types(dat, categorical_feature_cols, metric_feature_cols)

# Make predictions
predictions, _, _ = models.predict(dat=dat_features)

# Prepare predictions for DB
if write_db:
    df_for_db = dat_features.loc[:, ["date_time", "x", "y"]]
    df_for_db["model_id"] = model_id
    df_for_db["date_time_forecast"] = get_model_start_time()
    df_for_db["value"] = predictions
    df_for_db = df_for_db.loc[:, ["model_id", "date_time_forecast", "date_time", "x", "y", "value"]]

    # Convert date columns to unix format
    df_for_db.loc[:, "date_time"] = to_unix(df_for_db["date_time"])
    df_for_db.loc[:, "date_time_forecast"] = to_unix(df_for_db["date_time_forecast"])

    # Send results to the DB
    send_data_clickhouse(df=df_for_db, table_name="model_predictions_passive_samplers", mode="insert")
