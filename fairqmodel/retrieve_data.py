"""Retrieve feature data from ClickHouse for model training and prediction."""

from typing import Any

import pandas as pd
from clickhouse_connect.driver.exceptions import OperationalError
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from fairqmodel.db_connect import db_connect_source, db_connect_target, get_query
from fairqmodel.time_handling import get_current_local_time


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(60),
    retry=retry_if_exception_type(OperationalError),
    reraise=True,
)
def retrieve_data(
    mode: str,
    date_time_min: pd.Timestamp | None = None,
    date_time_max: pd.Timestamp | None = None,
    batch: int | None = None,
    include_future_data: bool = True,
    only_active_stations: bool = True,
) -> pd.DataFrame:
    """Retrieve all features for selected coordinates. Either for the stations or a given batch.

    :param mode: str, One of "stations", "grid", "grid_sim" and  "passive_samplers",
    specifies which tables are used for loading the data.
    :param date_time_min: Optional[pd.Timestamp], Minimal date to select
    :param date_time_max: Optional[pd.Timestamp], Maximal date to select
    :param batch: Optional[int], Batch number, pointing to a set of coordinates
    :param include_future_data: bool, If date_time_max is not set, it specifies
                                      whether predicted data is included or just observed data is returned.
                                      If date_time_max is set, this flag has no effect.
    :param only_active_stations: Optional[bool], indicates whether to return only the active stations, which are also
    used for prediction (default, only_active_stations=True), or all available for training (only_active_stations=False)

    :return: pd.DataFrame, Containing all features.
    """
    if mode not in ["stations", "grid", "passive_samplers", "grid_sim"]:
        raise ValueError("Mode must be one of: stations, grid, passive_samplers, grid_sim")

    logger.info("Loading data")

    coord_query_params: dict[str, Any] = {}
    if mode in ("stations", "passive_samplers"):
        coord_query = f"{mode}_coords"
        traffic_table_suffix = mode
        if mode == "stations":
            coord_query_params = {"only_active_stations": [True] if only_active_stations else [True, False]}
    elif mode in ("grid", "grid_sim"):
        coord_query = "coords_batches"
        traffic_table_suffix = "grid"
        assert batch is not None
        coord_query_params = {"batch": batch}
    else:
        raise ValueError("Mode must be one of: stations, grid, passive_samplers, grid_sim")

    filled_query = get_query("features", {"traffic_table_suffix": traffic_table_suffix})

    # If min and max datetime are not provided, select most recent and earliest possible date.
    date_time_min, date_time_max = fill_in_date_min_and_max(date_time_min, date_time_max, include_future_data)

    query_params = {
        "date_time_min": date_time_min,
        "date_time_max": date_time_max,
    }

    logger.info(f"Query params: date_time_min = '{date_time_min}', date_time_max = '{date_time_max}'")
    with db_connect_source() as db:
        db.execute(get_query(coord_query, {"mode": mode}), params=coord_query_params)
        db.execute(get_query("feature_station_avg"), params=query_params)
        dat = db.query_dataframe(filled_query, params=query_params)
    logger.info("Data successfully loaded\n")

    if dat.duplicated().any():
        logger.warning("Data contains duplicates")

    check_number_of_rows(batch, mode, dat, date_time_min, date_time_max)

    return dat


def fill_in_date_min_and_max(
    date_time_min: pd.Timestamp | None,
    date_time_max: pd.Timestamp | None,
    include_future_data: bool,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Fill in min and max date if they are None.

    :param date_time_min: Optional[pd.Timestamp], Minimal date to select
    :param date_time_max: Optional[pd.Timestamp], Maximal date to select
    :param include_future_data: bool, If date_time_max is not set, it specifies
                                      whether predicted data is included or just observed data is returned.
                                      If date_time_max is set, this flag has no effect.
    :return: Tuple[pd.Timestamp, pd.Timestamp], date_time_min and date_time_max
    """
    if date_time_max is None:
        td = pd.Timedelta(5, "day") if include_future_data else pd.Timedelta(0, "day")
        date_time_max = (pd.Timestamp(get_current_local_time()) + td).tz_convert(tz="UTC")
    else:
        date_time_max = date_time_max.tz_convert(tz="UTC")  # Convert to UTC, since the entries in the DB are UTC
    if date_time_min is None:
        date_time_min = pd.Timestamp("2015-01-01", tz="UTC")
    else:
        date_time_min = date_time_min.tz_convert(tz="UTC")  # Convert to UTC, since the entries in the DB are UTC
    return date_time_min, date_time_max


def check_number_of_rows(
    batch: int | None,
    mode: str | None,
    dat: pd.DataFrame,
    date_time_min: pd.Timestamp,
    date_time_max: pd.Timestamp,
) -> bool:
    """Check number of rows in the data under the assumption that hourly data is required for each pair of coordinates.

    Fails if there are not enough data points for a coordinate pair, including the case that the coordinates are
    completely absent.

    :param batch: Optional[int], Batch number, pointing to a set of coordinates
    :param mode: Optional[str], mode from retrieve_data, either "grid" or "grid_sim"
    :param dat: pd.DataFrame, Containing the data to be checked
    :param date_time_min: pd.Timestamp, Minimal date in the data
    :param date_time_max: pd.Timestamp, Maximal date in the data

    return: bool, true if ok, False if rows are missing
    """
    hours_expected = int(round(abs(date_time_max - date_time_min).total_seconds() / 3600, 0)) + 1

    if batch is None:
        n_coords = dat.groupby(["x", "y"]).ngroups
    else:
        with db_connect_source() as db:
            n_coords = int(
                db.query_dataframe(get_query("n_coords_batches", {"mode": mode}), params={"batch": batch}).n[0]
            )

    expected_rows = n_coords * hours_expected
    if expected_rows != len(dat):
        batch_msg = f" in batch {batch}" if batch is not None else ""
        msg = f"Expected Rows{batch_msg}: {expected_rows} do not match len(dat) = {len(dat)} - ok for training"
        logger.warning(msg)
        return False
    return True


def retrieve_cap_values() -> pd.DataFrame:
    """Retrieve cap values for outlier removal from the database."""
    with db_connect_target() as db:
        return db.query_dataframe(get_query("cap_values"))
