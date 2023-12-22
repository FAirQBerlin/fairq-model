import logging
from logging.config import dictConfig
from typing import Optional

from fairqmodel.db_connect import db_connect_target, get_query
from fairqmodel.extract_predict_load import extract_predict_load_grid
from fairqmodel.get_max_date_time import get_max_date_time
from fairqmodel.prediction_t_plus_k import get_model_settings
from fairqmodel.read_write_model_aux_functions import model_name_str
from fairqmodel.time_handling import get_model_start_time
from logging_config.logger_config import get_logger_config

dictConfig(get_logger_config())


def process_batch(msg, write_db: bool = False, mode: Optional[str] = "grid"):
    logging.info(f"Processing message {msg}")
    batch = int(eval(msg)["batch_id"])
    depvar = eval(msg)["depvar"]
    model_type = "spatial"

    if mode not in ["grid", "grid_sim"]:
        raise ValueError(f"Mode must be one of: grid, grid_sim, but is mode = {mode}")

    logging.info("Starting with depvar = {}, write_db = {}".format(depvar, write_db))

    # Retrieve selected model and settings
    with db_connect_target() as db:
        model_id = db.query_dataframe(get_query("final_model_id"), params={"model_type": model_type, "depvar": depvar})

    model_id = model_id.model_id[0]

    logging.info("Using model with model_id {}".format(model_id))

    query_params = {"model_type": model_name_str(model_type), "depvar": depvar}

    with db_connect_target() as db:
        available_models = db.query_dataframe(get_query("available_models"), params=query_params)

    # retrieve the trained model and its settings
    model_settings = get_model_settings(available_models, model_id)

    date_time_forecast = get_model_start_time(twice_daily=True)
    date_time_max = get_max_date_time(date_time_forecast)

    batch_report = extract_predict_load_grid(
        batch=batch,
        date_time_forecast=date_time_forecast,
        date_time_max=date_time_max,
        model_settings=model_settings,
        model_id=model_id,
        write_db=write_db,
        mode=mode,
    )

    logging.info(f"Finished process_batch batch: {batch}, depvar {depvar}")

    return batch_report
