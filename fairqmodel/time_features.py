"""Build time lag features for model training and prediction."""

import pandas as pd


def time_features(
    dat: pd.DataFrame,
    depvar: str,
    lags_actual: list[int] | None = None,
    lags_avg: list[int] | None = None,
) -> pd.DataFrame:
    """Add time features to DataFrame.

    :param dat: pd.DataFrame, Containing the data
    :param depvar: str, Dependent variable
    :param lags_actual: List, Actual lags
    :param lags_avg: List, Lags for average feature

    :return: DataFrame with time features
    """
    if lags_actual is None:
        lags_actual = []
    if lags_avg is None:
        lags_avg = []
    return time_lags(dat, depvar, lags_actual, lags_avg)


def time_lags(
    dat: pd.DataFrame,
    depvar: str,
    lags_actual: list[int] | None = None,
    lags_avg: list[int] | None = None,
) -> pd.DataFrame:
    """Add time lags for all stations to DataFrame.

    :param dat: pd.DataFrame, Containing the data
    :param depvar: str, Dependent variable
    :param lags_actual: List, Actual lags
    :param lags_avg: List, Lags for average feature

    :return: DataFrame with time lags
    """
    if lags_actual is None:
        lags_actual = []
    if lags_avg is None:
        lags_avg = []
    lags_all = sorted(set(lags_actual + lags_avg))

    if len(lags_all) > 0:
        # Create the columns for lags of both types (i.e. lags_actual and lags_avg)
        dat = dat.groupby("station_id").apply(lambda x: get_station_lags(x, depvar, lags_all), include_groups=False)
        dat = dat.copy()
        dat["station_id"] = dat.index.get_level_values(0)
        cols = dat.columns.tolist()
        # Reorder columns to have the station_id at the beginning
        dat = dat[cols[-1:] + cols[:-1]]
        dat = dat.reset_index("station_id", drop=True)

        # If there are values in lags_avg, construct the average feature
        if len(lags_avg) > 0:
            cols = [f"{depvar}_lag{x}" for x in lags_avg]
            col_name = f"lag_avg_{sorted(lags_avg)}".replace("[", "(").replace("]", ")")
            dat.loc[:, col_name] = dat.loc[:, cols].astype(float).mean(axis=1)

            if f"{depvar}_train" in dat.columns:
                cols = [f"{depvar}_lag{x}_train" for x in lags_avg]
                col_name = f"lag_avg_{sorted(lags_avg)}_train".replace("[", "(").replace("]", ")")
                dat.loc[:, col_name] = dat.loc[:, cols].mean(axis=1)

    return dat


def get_station_lags(
    dat: pd.DataFrame,
    depvar: str,
    lags: list[int],
) -> pd.DataFrame:
    """Add time lags to DataFrame.

    :param dat: pd.DataFrame, Contains data of one station
    :param depvar: str, Dependent variable
    :param lags: List, Lags to add

    :return: DataFrame with time lags
    """
    # sort dat by date_time descending

    dat = dat.sort_values("date_time")

    # Check if a separated depvar column for the training exists
    requires_train_lags = f"{depvar}_train" in dat.columns

    for lag in lags:
        dat.loc[:, f"{depvar}_lag{lag}"] = dat.loc[:, depvar].astype(float).shift(lag)

        # If a separated depvar column for the training exists, lags are built accordingly
        if requires_train_lags:
            dat.loc[:, f"{depvar}_lag{lag}_train"] = dat.loc[:, f"{depvar}_train"].shift(lag)

    return dat
