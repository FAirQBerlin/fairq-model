import json
import logging
from logging.config import dictConfig
from typing import Optional
from uuid import uuid4

from fairqmodel.db_connect import db_connect_source, get_query
from logging_config.logger_config import get_logger_config

dictConfig(get_logger_config())


def receipt(err, msg):
    """Get a success receipt when producing kafka events
    :param err error
    :param msg message
    """
    if err is not None:
        print("Error: {}".format(err))
    else:
        message = "Produced on topic {}, partition {}, value {}, key {}\n".format(
            msg.topic(), msg.partition(), msg.value().decode("utf-8"), msg.key().decode("utf-8")
        )
        logging.info(message)


def produce_events(topic, p, mode: Optional[str] = "grid"):
    """Produce kafka events for the variables no2, pm10, and pm25 on all batches in the coords_batches table
    :param topic to produce events to
    :param p initiated kafka producer
    :param mode mode to produce the predictions for - used to define max_batches
    """
    if mode not in ["grid", "grid_sim"]:
        raise ValueError(f"Mode must be one of: grid, grid_sim, but is mode = {mode}")

    n_partitions = len(p.list_topics().topics[topic].partitions)

    with db_connect_source() as db:
        max_batch = db.query_dataframe(get_query("max_batch", {"mode": mode})).max_batch[0]

    for i in range(1, max_batch + 1):
        for depvar in ["no2", "pm10", "pm25"]:
            key = str(uuid4())
            event = {"key": key, "batch_id": i, "depvar": depvar}
            m = json.dumps(event)
            p.poll(0.1)
            p.produce(
                topic=topic,
                value=m.encode("utf-8"),
                key=str(i % n_partitions),
                callback=receipt,
            )
            p.flush()
