import logging
from logging.config import dictConfig
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fairqmodel.model_wrapper import ModelWrapper
from logging_config.logger_config import get_logger_config

dictConfig(get_logger_config())


def make_lag_adjusted_prediction(
    fold: dict,
    models: ModelWrapper,
    feature_cols: List[str],
    depvar: str,
    lags_actual: List[int],
    lags_avg: List[int],
    calc_metrics: bool = False,
    testing: bool = False,
) -> Tuple[pd.DataFrame, Optional[float]]:
    """Evaluates the lag adjusted predictions for a given fold.

    :param fold: dic, Contains information of the fold
    :param models: ModelWrapper, Wrapper object containing one or two models
    :param feature_cols: List, Contains variable names
    :param lags_actual: List, Actual lags
    :param lags_avg: List, Lags for average feature
    :param calc_metrics, bool, Specifies if metrics will be printed for predictions
    :param testing, bool, if True, all columns are returned

    :return: pd.DataFrame, Containing all predictions
    :return: Optional[float], RMSE value of this fold
    """
    # Get the data for which the predictions are performed
    dat = fold["test"]

    # Get lag adjusted prediction
    fold_dat_for_loop, time_points = prepare_fold_test_dat(dat)

    # Always predict only one time point, and then use that prediction as the lag value for the next time point
    loop_results: list[dict] = []

    for loop_index, time_point in enumerate(time_points):
        time_point_prediction = get_lag_adjusted_prediction(
            fold_dat_for_loop,
            time_point,
            models,
            feature_cols,
            depvar,
            loop_index,
            lags_actual,
            lags_avg,
            loop_results,
        )
        loop_results.append(time_point_prediction)
    output_cols = ["station_id", "date_time", depvar, "pred"]
    all_results = pd.concat([x["pred"] for x in loop_results]).sort_values(["station_id", "date_time"])

    if not testing:
        all_results = all_results.loc[:, output_cols]

    all_results.loc[:, "date_time_forecast"] = fold["ts_fold_max_train_date"]

    metric_value = None
    if calc_metrics:
        predictions = all_results.pred
        label = all_results.loc[:, depvar]

        rmse = mean_squared_error(label, predictions, squared=False)
        mae = mean_absolute_error(label, predictions)
        r2 = r2_score(label, predictions)

        logging.info(f"{rmse=:.3f}\t{mae=:.3f}\t{r2=:.3f} \n")
        metric_value = rmse

    return all_results, metric_value


def prepare_fold_test_dat(fold_dat_test: pd.DataFrame):
    """Prepare the test data for the time series model.

    Args:
        fold_dat_test: pandas.DataFrame, test data

    Returns:
        pandas.DataFrame, the test data for the time series model
    """
    # TODO: duplicates should have been solved previously
    fold_dat_test_for_loop = fold_dat_test.copy(deep=True).drop_duplicates(subset=["station_id", "date_time"])
    test_time_points = np.sort(fold_dat_test.date_time.unique())
    return fold_dat_test_for_loop, test_time_points


def get_lag_adjusted_prediction(
    fold_dat_test_for_loop: pd.DataFrame,
    time_point: str,
    models: ModelWrapper,
    feature_cols: List[str],
    depvar: str,
    loop_index: int,
    lags_actual: List[int],
    lags_avg: List[int],
    loop_results: List,
) -> dict:
    """Get the lag adjusted prediction.

    Args:
        fold_dat_test_for_loop: pandas.DataFrame, test data
        time_point: str, time point to filter the test data
        models: ModelWrapper, Wrapper object containing one or two models
        feature_cols: List[str] of str, feature columns
        depvar: str, dependent variable
        loop_index: int, loop index used to determine the lag correction
        lags_actual: List, Actual lags
        lags_avg: List, Lags for average feature
        loop_results: List, loop results

    Returns:
        pandas.DataFrame, the lag adjusted prediction

    """
    # Get the row to use fore prediction
    time_point_filter = fold_dat_test_for_loop.date_time == time_point
    row_for_pred = fold_dat_test_for_loop.loc[time_point_filter, :].copy()

    # If we are at loop_index > 0, we need replace the lag value with the prediction
    # of the previous iteration
    if loop_index > 0:
        row_for_pred = fill_lags_with_previous_predictions(
            row_for_pred,
            lags_actual,
            lags_avg,
            loop_index,
            loop_results,
            depvar,
        )

    # Do the prediction with the fixed dataset
    pred_output, _, _ = models.predict(dat=row_for_pred)
    row_for_pred.loc[:, "pred"] = pred_output

    return {"loop_index": loop_index, "time_point": time_point, "pred": row_for_pred}


def fill_lags_with_previous_predictions(
    row_for_pred: pd.DataFrame,
    lags_actual: List[int],
    lags_avg: List[int],
    loop_index: int,
    loop_results: List[dict],
    depvar: str,
):
    """Fill the lags with the previous predictions.

    Args:
        row_for_pred: pandas.DataFrame, the row to use for prediction
        lags_actual: List, Actual lags
        lags_avg: List, Lags for average feature
        loop_index: int, loop index used to determine the lag correction
        loop_results: List, loop results
        depvar: str, dependent variable

    Returns:
        pandas.DataFrame, the row to use for prediction
    """
    lags_all = sorted(list(set(lags_actual + lags_avg)))

    # Update both types of lags
    for lag in lags_all:
        # Replace each lag with the predicted values
        # The loop index is the number of hours after the first hour in the test data.
        # For example, if the test data starts at 15:00, then the entry for 16:00 has loop index 1.
        # If the time is far enough in the future, we need to replace the lag variables with the respective prediction.
        # For example: For 18:00 and later, we need to replace lag 3, because 18-3=15, which is already in the test
        # data.
        if loop_index >= lag:
            row_for_pred = overwrite_lags_with_prev_predictions(
                row_for_pred,
                lag,
                loop_results,
                depvar,
            )
    # Update the average feature (if it exists)
    if len(lags_avg) > 0:
        cols = [f"{depvar}_lag{x}" for x in lags_avg]
        col_name = f"lag_avg_{sorted(lags_avg)}".replace("[", "(").replace("]", ")")
        row_for_pred.loc[:, col_name] = row_for_pred.loc[:, cols].mean(axis=1)

    return row_for_pred


def overwrite_lags_with_prev_predictions(
    row_for_pred: pd.DataFrame,
    lag: int,
    loop_results: List[dict],
    depvar: str,
):
    """Overwrite the lags with the previous predictions.

    Args:
        row_for_pred: pandas.DataFrame, the row to use for prediction
        lag: int, lag that is used in the time series model
        loop_results: List, loop results
        depvar: str, dependent variable

    Returns:
        pandas.DataFrame, the row_for_pred with the lag replaced with the previous predictions
    """
    # Subtract lag hours from current_time_point to find the past prediction to overwrite the lag value
    current_time_point = row_for_pred.date_time.iloc[0]
    lag_time_point = pd.to_datetime(current_time_point) - pd.Timedelta(hours=lag)
    loop_lag = [past_pred for past_pred in loop_results if past_pred["time_point"] == lag_time_point]

    # Get the data from the lagged loop iteration
    loop_lag_prediction = loop_lag[0]["pred"].loc[:, ["station_id", "pred"]]

    # Overwrite the lag value with the prediction
    row_for_pred = row_for_pred.merge(loop_lag_prediction, how="left", on="station_id")
    row_for_pred.loc[:, f"{depvar}_lag{lag}"] = row_for_pred.pred
    row_for_pred.drop(columns=["pred"], inplace=True)
    return row_for_pred


def get_eval_metrics(loop_results: List[dict], depvar: str) -> pd.DataFrame:
    """Get the evaluation metrics.

    Args:
        loop_results: List, loop results
        depvar: str, dependent variable

    Returns:
        dict, the evaluation metrics for the time series model
    """

    eval_metrics = []
    all_data_joint = []
    for loop_res in loop_results:
        tmp = loop_res["pred"].loc[:, ["date_time", "station_id", depvar, "pred"]]

        y_actual = tmp.loc[:, depvar]
        y_predicted = tmp.loc[:, "pred"]
        eval_metrics.append(
            {
                "time_point": loop_res["time_point"],
                "mae": mean_absolute_error(y_actual, y_predicted),
                "rmse": mean_squared_error(y_actual, y_predicted, squared=False),
                "r_squared": r2_score(y_actual, y_predicted),
            },
        )
        all_data_joint.append(tmp)
    eval_metrics_by_timestep = pd.DataFrame(eval_metrics)

    all_data_joint_df = pd.concat(all_data_joint)
    y_all_actual = all_data_joint_df.loc[:, depvar]
    y_all_predicted = all_data_joint_df.loc[:, "pred"]
    eval_metrics_all = pd.DataFrame(
        {
            "time_point": "all",
            "mae": mean_absolute_error(y_all_actual, y_all_predicted),
            "rmse": mean_squared_error(y_all_actual, y_all_predicted, squared=False),
            "r_squared": r2_score(y_all_actual, y_all_predicted),
        },
        index=[0],
    )

    return pd.concat([eval_metrics_all, eval_metrics_by_timestep])
