import logging
from json import loads
from logging.config import dictConfig
from typing import List

import pandas as pd

from fairqmodel.build_splits import prepare_folded_input
from fairqmodel.db_connect import send_data_clickhouse
from fairqmodel.model_wrapper import ModelWrapper
from fairqmodel.prediction_lag_adjusted import make_lag_adjusted_prediction
from fairqmodel.read_write_model_db import retrieve_model_from_db
from fairqmodel.time_handling import to_unix
from logging_config.logger_config import get_logger_config

dictConfig(get_logger_config())


def prediction_t_plus_k(
    dat: pd.DataFrame,
    models: ModelWrapper,
    model_id: int,
    feature_cols: list[str],
    depvar: str,
    lags_actual: List[int],
    lags_avg: List[int],
    t_plus_k_params: dict,
    table_name: str,
    write_db=False,
    verbose=False,
    calc_metrics: bool = False,
    include_current_time_point: bool = False,
) -> None:
    """Performs the prediction for specified time points 't', each for the upcoming number of hours 'k'.
    Note: This function is similar to the t_plus_k CV and uses similar terminology for consistency.
    :param dat: pd.DataFrame, Data to perform the predictions on
    :param models: ModelWrapper, Wrapper object containing one or two models
    :param model_id: int, Id under which the used model is accessible in the DB
    :param feature_cols: list[str], Variable names used for predictions
    :param depvar: str, Name of the dependent variable
    :param lags_actual: List[int], Actual lags
    :param lags_avg: List[int], Lags for average feature
    :param t_plus_k_params: dict, Keys of this dict:
                            "n_windows": number of time points called 't',
                            "n_train_years": not used for prediction
                            "window_size": range of each prediction window called 'k',
                            "step_size": how much each 't' is shifted against the previous one.
                            "prediction_hour": Specifies the last hour of training,
                                               i.e. the time point where the prediction is made
    :param table_name: str, Specifies the DB table to which the results are written:
                        Future (script 3c): "model_predictions_temporal"
                        Past (script 3b): e.g. "model_predictions_temporal_tweak_values"
    :param write_db: bool, Specifies if the predictions are written to the DB
    :param verbose: bool, Specifies if the progress is logged
    :param calc_metrics: bool, Specifies if quality metrics are calculated
    :param include_current_time_point: bool, Specifies if a prediction for the current time point (k=0) is made.
                                             Default is False

    :return: None
    """
    time_folds = prepare_folded_input(
        dat,
        n_cv_windows=t_plus_k_params["n_windows"],
        n_train_years=t_plus_k_params["n_train_years"],
        test_cv_window_size=t_plus_k_params["window_size"],
        step_size=t_plus_k_params["step_size"],
        prediction_hour=t_plus_k_params["prediction_hour"],
        include_current_time_point=include_current_time_point,
    )

    for fold in time_folds:
        if verbose:
            logging.info(f"Currently predicting fold {fold['ts_fold_id']}/{len(time_folds)}")
            logging.info(f"date_time_forecast: {fold['ts_fold_max_train_date'].strftime('%Y-%m-%d %H:%M:%S')}")

        all_results, _ = make_lag_adjusted_prediction(
            fold, models, feature_cols, depvar, lags_actual, lags_avg, calc_metrics
        )

        # Write predictions to DB
        if write_db:
            df_predictions = all_results.rename(columns={"pred": "value"}).drop(columns=[depvar])
            df_predictions["model_id"] = model_id

            # Reorder columns
            df_predictions = df_predictions[["model_id", "date_time_forecast", "date_time", "station_id", "value"]]
            # Convert date_time to unix
            df_predictions.loc[:, "date_time"] = to_unix(df_predictions["date_time"])
            df_predictions.loc[:, "date_time_forecast"] = to_unix(df_predictions["date_time_forecast"])

            send_data_clickhouse(
                df=df_predictions,
                table_name=table_name,
                mode="replace",
            )


def get_model_settings(available_models: pd.DataFrame, MODEL_ID: int) -> dict:
    """Gets the settings associated with the selected model_id.

    :param available_models: pd.DataFrame, Containing all currently available models
    :param MODEL_ID: int, Id of the selected model

    :return: dict, Containing relevant settings, i.e. model_object, variable names,
                lags and the name of the dependent variable

    """
    assert MODEL_ID in available_models.model_id.unique(), "Selected model id not available."

    # Prepare objects to make predictions
    models = retrieve_model_from_db(MODEL_ID)
    assert models.model_1 is not None, "Retrieved first stage is not set correctly"
    assert models.model_1.feature_names is not None
    assert models.model_1.feature_types is not None
    feature_cols = models.model_1.feature_names
    feature_types = models.model_1.feature_types
    if models.use_two_stages:
        assert models.model_2 is not None, "Retrieved second stage is not set correctly"
        assert models.model_2.feature_names is not None
        assert models.model_2.feature_types is not None
        feature_cols.extend(models.model_2.feature_names)
        feature_types.extend(models.model_2.feature_types)

    categorical_feature_cols = [
        feature for feature, feature_type in zip(feature_cols, feature_types) if feature_type == "c"
    ]
    metric_feature_cols = [feature for feature, feature_type in zip(feature_cols, feature_types) if feature_type != "c"]

    description_model_1 = loads(available_models.loc[available_models.model_id == MODEL_ID, "description"].values[0])

    lags = eval(description_model_1["lags"])
    lags_avg = eval(description_model_1["lags_avg"]) if "lags_avg" in description_model_1.keys() else []

    if models.use_two_stages:
        description_model_2 = loads(
            available_models.loc[available_models.model_id == MODEL_ID, "description_residuals"].values[0]
        )
        lags.extend(eval(description_model_2["lags"]))
        if "lags_avg" in description_model_2.keys():
            lags_avg.extend(eval(description_model_2["lags_avg"]))

    max_training_date_model_1 = description_model_1["training_period"][1]
    max_training_date_model_2 = description_model_2["training_period"][1] if models.use_two_stages else None

    model_settings = {
        "models": models,
        "feature_cols": list(set(feature_cols)),
        "categorical_feature_cols": list(set(categorical_feature_cols)),
        "metric_feature_cols": list(set(metric_feature_cols)),
        "lags": lags,
        "lags_avg": lags_avg,
        "depvar": models.depvar,
        "max_training_date_model_1": max_training_date_model_1,
        "max_training_date_model_2": max_training_date_model_2,
    }

    return model_settings
