import math
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd

from fairqmodel.notebook_helpers.limit_values import aggregate_data_daily


def test_aggregate_data_daily():
    # Arrange
    depvar = "no2"
    n_days = 2
    n_stations = 2
    dat_size = n_stations * n_days * 24

    # Each day has to start at 1am and ends with 0am of the following day
    target_hours = np.array(list(np.arange(1, 24)) + [0])

    # Build columns of DataFrame
    station_id = ["3141"] * (dat_size // n_stations) + ["2718"] * (dat_size // n_stations)
    pred = np.random.rand(dat_size)
    obs = np.random.rand(dat_size)
    date_time = (
        np.arange(datetime(2021, 1, 1), datetime(2022, 1, 1), timedelta(hours=1))
        .astype(datetime)[: (dat_size // n_stations)]
        .tolist()
        * n_stations
    )

    dat = pd.DataFrame({"station_id": station_id, "pred": pred, "no2": obs, "date_time": date_time})

    # Act
    pred_vs_obs, dat = aggregate_data_daily(dat.copy(deep=True))
    pred_list, obs_list, hour_list = verify_resampling(dat, pred_vs_obs, depvar, target_hours)

    # Assert
    assert all(obs_list)
    assert all(pred_list)
    assert all(hour_list)


def verify_resampling(
    dat: pd.DataFrame, pred_vs_obs: pd.DataFrame, depvar: str, target_hours: np.ndarray
) -> Tuple[List[bool], List[bool], List[bool]]:
    """Checks if in the resampling step the dates have been aggregated as intended"""

    abs_tol = 0.0001  # Tolerance for float comparison

    pred_list = []
    obs_list = []
    hour_list = []

    for station_id in pred_vs_obs.station_id.unique():
        # The last date is excluded from this loop since there is no data for the upcoming day
        for date in pred_vs_obs.date_time.to_list()[:-1]:
            next_date = str(pd.Timestamp(date) + pd.Timedelta(1, "day"))

            dat_day = dat.query(
                f"station_id == '{station_id}' and date_time > '{str(date)}' and date_time <='{next_date}'"
            )

            target_obs, target_pred = dat_day[[depvar, "pred"]].mean().values
            hours = dat_day.date_time.dt.hour.values

            res_obs, res_pred = pred_vs_obs.query(f"station_id == '{station_id}' and date_time == '{date}'")[
                [depvar, "pred"]
            ].values[0]

            # NOTE: Strict comparison 'a == b' leads to a mistake due to precision
            # It is sufficient if predictions and observations are equal up to the fourth decimal
            pred_list.append(math.isclose(target_pred, res_pred, abs_tol=abs_tol))
            obs_list.append(math.isclose(target_obs, res_obs, abs_tol=abs_tol))
            hour_list.append((hours == target_hours).all())

    return pred_list, obs_list, hour_list
