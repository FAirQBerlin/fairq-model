import logging
from logging.config import dictConfig
from typing import List, Optional

import pandas as pd

from fairqmodel.retrieve_data import retrieve_cap_values
from logging_config.logger_config import get_logger_config

dictConfig(get_logger_config())


def fix_column_types(
    dat: pd.DataFrame,
    categorical_feature_cols: List[str],
    metric_feature_cols: List[str],
) -> pd.DataFrame:
    """Changes the dtype of variables to either float or category.
    This is required as a pre-processing step, because some
    variables are stored in a different dtype in the DB.
    The date_time column is converted to type date_time.

    :param dat: DataFrame, containing all variables plus a date_time column
    :param categorical_feature_cols: List, names of categorical variables
    :param metric_feature_cols: List, names of metric variables

    :return: pd.DataFrame, DataFrame with identical content,
    but each variable is either float or category.
    """
    # NOTE: The train lags are not included in the feature_cols list, hence:
    train_lags = [col for col in dat.columns if "_train" in col]
    all_metric_cols = metric_feature_cols + train_lags

    dat = dat.astype({x: "category" for x in categorical_feature_cols})
    dat = dat.astype({x: "float" for x in all_metric_cols})
    dat.date_time = pd.to_datetime(dat.date_time)

    return dat


def drop_stations_without_this_depvar(dat: pd.DataFrame, depvar: str) -> pd.DataFrame:
    """
    Drop all stations that don't deliver any value for the selected dependent variable.
    Predictions should only be made if a station delivers the variable because otherwise the target value and the lags
    will be unknown.
    It's only checked if any value is given, not if all values in the past are given. There should not be any gaps in
    the data anymore because they have been filled in the gap filling, which must not be tested here.
    :param dat: pd.DataFrame, Input data for model predictions
    :param depvar: str, Dependent variable, can be "no2", "pm10", or "pm25"
    :return: pd.DataFrame, Comprised of the same columns and maybe fewer rows
    """
    any_depvar_value = dat.groupby("station_id").apply(lambda x: any(x[depvar].notnull()))
    stations_with_this_depvar = any_depvar_value.index[any_depvar_value]
    dat = dat.loc[dat.station_id.isin(stations_with_this_depvar)]
    return dat


def cap_outliers(dat: pd.DataFrame, cap_values: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Extreme outliers are removed from the dependent variables by capping them at the given values.
    The default cap values are the 0.999 quantiles from the data, calculated on the data from [2015-01-01, 2023-01-01].
    If no cap values are specified, they are accessed by the retrieve_cap_values() function from the DB
    (i.e. from db table fairq_(prod_)output.cap_values)

    :param dat: pd.DataFrame, Containing the data
    :param cap_values: Optional[pd.DataFrame], Can be used to specify non-default values for outlier removal

    :return:pd.DataFrame
    """

    # Access cap values
    if cap_values is None:
        cap_values = retrieve_cap_values()

    cap_no2 = cap_values.query("pollutant == 'no2'").cap_value_outlier.item()
    cap_pm10 = cap_values.query("pollutant == 'pm10'").cap_value_outlier.item()
    cap_pm25 = cap_values.query("pollutant == 'pm25'").cap_value_outlier.item()

    # Identify values above the cap values
    outliers_no2 = dat.no2 > cap_no2
    outliers_pm10 = dat.pm10 > cap_pm10
    outliers_pm25 = dat.pm25 > cap_pm25

    # Remove outliers from selected columns
    dat.loc[outliers_no2, "no2"] = cap_no2
    dat.loc[outliers_pm10, "pm10"] = cap_pm10
    dat.loc[outliers_pm25, "pm25"] = cap_pm25

    logging.info(
        "Number of outliers capped: no2: {}, pm10: {}, pm25: {}".format(
            outliers_no2.values.sum(), outliers_pm10.values.sum(), outliers_pm25.values.sum()
        )
    )

    return dat


def cap_high_values(dat: pd.DataFrame, depvar: str, cap_values: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Caps all columns corresponding to the specified depvar (the depvar column itself and its lags)
    to allow for more stable training.
    NOTE: The goal of this function is to cap unreasonable large input values before model training.
          As this step is performed after the lags have been constructed, the lag_avg feature was calculated from
          un-capped values. If the lag_avg in total is above the cap_value, it is capped. However, it is not capped
          in case that the total lag_avg is the below the cap_value, but some of its parts exceeded it.

    :param dat: pd.DataFrame, Containing the (uncapped) data
    :param depvar: str, Name of the target variable
    :param cap_values: Optional[pd.DataFrame], Can be used to specify non-default values for outlier removal

    :return: pd.DataFrame, Containing the capped data
    """
    # Access cap values
    if cap_values is None:
        cap_values = retrieve_cap_values()

    # Select all columns viable for capping
    # NOTE: cams and cams-lags are not affected by the capping
    cap_columns = [col for col in dat.columns if (f"{depvar}_lag" in col) or ("lag_avg" in col)] + [depvar]

    # Access cap values
    if cap_values is None:
        cap_values = retrieve_cap_values()

    cap_val = cap_values.query(f"pollutant == '{depvar}'").cap_value_training.item()

    # Cap all columns corresponding to the depvar
    for cap_column in cap_columns:
        outliers = dat[cap_column] > cap_val
        dat.loc[outliers, cap_column] = cap_val

    return dat
