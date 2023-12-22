from typing import Dict, List

import pandas as pd


def get_station_cv_folds(dat: pd.DataFrame) -> List[Dict]:
    """Create folds for spatial CV.
    For each fold one station is left out.

    :param dat: pd.DataFrame, DataFrame with all variables and station_ids

    :return: List[Dict], List with one dictionary per fold,
    containing station_id, train and test.
    """
    station_cv_folds = []
    for station_id in dat.station_id.unique():
        fold = dict()
        fold["station_id"] = station_id
        fold["train"] = dat.loc[dat.station_id != station_id, :].index.values.tolist()
        fold["test"] = dat.loc[dat.station_id == station_id, :].index.values.tolist()
        station_cv_folds.append(fold)

    return station_cv_folds
