"""Produce JSON batch events for grid prediction jobs."""

import json
from pathlib import Path

from loguru import logger

from fairqmodel.db_connect import db_connect_source, get_query


def produce_json_events(model_type: str | None = "grid") -> None:
    """Produce JSON events for the variables no2, pm10, and pm25 on all batches in the coords_batches table.

    :param model_type model_type to produce the predictions for - used to define max_batches
    """
    if model_type not in ["grid", "grid_sim"]:
        raise ValueError(f"Mode must be one of: grid, grid_sim, but is model_type = {model_type}")

    with db_connect_source() as db:
        max_batch = db.query_dataframe(get_query("max_batch", {"mode": model_type})).max_batch[0]

    events = []
    for i in range(1, max_batch + 1):
        for depvar in ["no2", "pm10", "pm25"]:
            event = {"batch_id": i, "depvar": depvar}
            events.append(event)
            logger.info(f"Produced event: {event}")

    with Path("/tmp/events.json").open("w") as file:  # we can potentially remove the mode argument
        json.dump(events, file, ensure_ascii=False)
