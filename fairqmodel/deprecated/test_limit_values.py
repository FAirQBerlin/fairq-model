import math

import numpy as np
import pandas as pd


def verify_resampling(
    dat: pd.DataFrame, pred_vs_obs: pd.DataFrame, depvar: str, target_hours: np.ndarray
) -> tuple[list[bool], list[bool], list[bool]]:
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
                f"station_id == '{station_id}' and date_time > '{date!s}' and date_time <='{next_date}'"
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
