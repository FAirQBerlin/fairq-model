import json
import logging
from logging.config import dictConfig
from typing import Optional

from fairqmodel.db_connect import db_connect_source, get_query
from logging_config.logger_config import get_logger_config

dictConfig(get_logger_config())


def produce_json_events(model_type: Optional[str] = "grid") -> None:
    """Produce json events for the variables no2, pm10, and pm25 on all batches in the coords_batches table
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
            logging.info(f"Produced event: {event}")

    with open("/tmp/events.json", "w") as file:  # we can potentially remove the mode argument
        json.dump(events, file, ensure_ascii=False)
