"""Perform t+k predictions over multiple time folds with optional DB writes."""

import ast
from json import loads

import pandas as pd
from loguru import logger

from fairqmodel.build_splits import prepare_folded_input, slice_data_frame
from fairqmodel.db_connect import send_data_clickhouse
from fairqmodel.model_wrapper import ModelWrapper
from fairqmodel.prediction_lag_adjusted import make_lag_adjusted_prediction
from fairqmodel.read_write_model_db import retrieve_model_from_db


def prediction_t_plus_k(
    dat: pd.DataFrame,
    models: ModelWrapper,
    model_id: int,
    feature_cols: list[str],  # noqa: ARG001 - kept for API compatibility; model derives features internally
    depvar: str,
    lags_actual: list[int],
    lags_avg: list[int],
    t_plus_k_params: dict,
    table_name: str,
    write_db=False,
    verbose=False,
    calc_metrics: bool = False,
    include_current_time_point: bool = False,
) -> None:
    """Perform the prediction for specified time points 't', each for the upcoming number of hours 'k'.

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
        fold["test"] = slice_data_frame(
            dat,
            lower_bound=fold["test_window_cut_min_modified"],
            upper_bound=fold["test_window_cut_max"],
        )

        fold["train"] = slice_data_frame(
            dat,
            lower_bound=fold["train_window_cut_min"],
            upper_bound=fold["test_window_cut_min_modified"],
        )
        if verbose:
            logger.info(f"Currently predicting fold {fold['ts_fold_id']}/{len(time_folds)}")
            logger.info(f"date_time_forecast: {fold['ts_fold_max_train_date'].strftime('%Y-%m-%d %H:%M:%S')}")

        all_results, _ = make_lag_adjusted_prediction(fold, models, depvar, lags_actual, lags_avg, calc_metrics)

        # Write predictions to DB
        if write_db:
            df_predictions = all_results.rename(columns={"pred": "value"}).drop(columns=[depvar])
            df_predictions["model_id"] = model_id

            # Reorder columns
            df_predictions = df_predictions[["model_id", "date_time_forecast", "date_time", "station_id", "value"]]

            send_data_clickhouse(
                df=df_predictions,
                table_name=table_name,
                mode="insert",
            )


def get_model_settings(model_id: int) -> dict:
    """Get the settings associated with the selected model_id.

    :param model_id: int, Id of the selected model

    :return: dict, Containing relevant settings, i.e. model_object, variable names,
                lags and the name of the dependent variable

    """
    # Prepare objects to make predictions
    models = retrieve_model_from_db(model_id)
    assert models.model_1 is not None, "Retrieved first stage is not set correctly"
    assert models.model_1.feature_names is not None
    assert models.model_1.feature_types is not None
    feature_cols = list(models.model_1.feature_names)
    feature_types = list(models.model_1.feature_types)
    if models.use_two_stages:
        assert models.model_2 is not None, "Retrieved second stage is not set correctly"
        assert models.model_2.feature_names is not None
        assert models.model_2.feature_types is not None
        feature_cols.extend(models.model_2.feature_names)
        feature_types.extend(models.model_2.feature_types)

    categorical_feature_cols = [
        feature for feature, feature_type in zip(feature_cols, feature_types, strict=False) if feature_type == "c"
    ]
    metric_feature_cols = [
        feature for feature, feature_type in zip(feature_cols, feature_types, strict=False) if feature_type != "c"
    ]

    if models.description is None:
        raise ValueError(f"No description metadata found for model_id {model_id}")
    description_model_1 = loads(models.description)

    lags = ast.literal_eval(description_model_1["lags"])
    lags_avg = ast.literal_eval(description_model_1["lags_avg"]) if "lags_avg" in description_model_1 else []
    if models.use_two_stages:
        assert models.description_residuals is not None, (
            f"No residuals description metadata found for model_id {model_id}"
        )
        description_model_2 = loads(models.description_residuals)
        lags.extend(eval(description_model_2["lags"]))
        if "lags_avg" in description_model_2:
            lags_avg.extend(eval(description_model_2["lags_avg"]))

    max_training_date_model_1 = description_model_1["training_period"][1]
    max_training_date_model_2 = description_model_2["training_period"][1] if models.use_two_stages else None

    return {
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
