"""Process a single grid prediction batch message end-to-end."""

from loguru import logger

from fairqmodel.db_connect import db_connect_target, get_query
from fairqmodel.extract_predict_load import extract_predict_load_grid
from fairqmodel.get_max_date_time import get_max_date_time
from fairqmodel.prediction_t_plus_k import get_model_settings
from fairqmodel.time_handling import get_model_start_time


# debugging with msg = '{"batch_id": 1, "depvar": "no2"}'; mode = "grid"; write_db = False
def process_batch(msg, write_db: bool = False, mode: str | None = "grid"):
    """Process a single prediction batch message and return the batch report.

    :param msg: str or dict, message containing batch_id and depvar
    :param write_db: bool, whether to write predictions to the database
    :param mode: str, one of "grid" or "grid_sim"

    :return: dict, batch report with batch id and success status
    """
    logger.info(f"Processing message {msg}")
    batch = int(eval(msg)["batch_id"])
    depvar = eval(msg)["depvar"]
    model_type = "spatial"

    if mode not in ["grid", "grid_sim"]:
        raise ValueError(f"Mode must be one of: grid, grid_sim, but is mode = {mode}")

    logger.info(f"Starting with depvar = {depvar}, write_db = {write_db}")

    # Retrieve selected model and settings
    with db_connect_target() as db:
        model_id = db.query_dataframe(get_query("final_model_id"), params={"model_type": model_type, "depvar": depvar})

    model_id = model_id.model_id[0]

    logger.info(f"Using model with model_id {model_id}")

    # retrieve the trained model and its settings
    model_settings = get_model_settings(model_id)

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

    logger.info(f"Finished process_batch batch: {batch}, depvar {depvar}")

    return batch_report
