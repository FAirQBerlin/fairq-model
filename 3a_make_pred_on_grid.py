"""
This script performs spatial predictions without lags of the dependent variable.
A pre-trained model is loaded from the DB and performs the prediction on the grid.
Due to computational constraints the predictions are performed for few x-coordinates at a time.
The predictions can be written to the DB.
"""

import logging
from logging.config import dictConfig

from fairqmodel.command_line_args import get_command_args
from fairqmodel.db_connect import db_connect_source, db_connect_target, get_query
from fairqmodel.extract_predict_load import extract_predict_load_grid, get_batches_to_reschedule
from fairqmodel.get_max_date_time import get_max_date_time
from fairqmodel.prediction_t_plus_k import get_model_settings
from fairqmodel.read_write_model_aux_functions import model_name_str
from fairqmodel.time_handling import get_model_start_time
from logging_config.logger_config import get_logger_config

dictConfig(get_logger_config())

# Configuration
DEV = False

verbose = True

model_type = "spatial"  # "temporal", "spatial"

# passed command args sample: ['filename.py', 'depvar=no2', 'write_db=True']
depvar = get_command_args("depvar") or "no2"
write_db = get_command_args("write_db") or False

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

reporting = list()


with db_connect_source() as db:
    max_batch = db.query_dataframe(get_query("max_batch")).max_batch[0]

logging.info(f"Starting with extract, predict and load for {max_batch} batches.")

batches_to_process = range(1, max_batch + 1)
n_runs = 2

for run in range(n_runs):
    if run > 0:
        logging.info(f"re-scheduling batches: {batches_to_process}")

    for batch in batches_to_process:
        batch_report = extract_predict_load_grid(
            batch=batch,
            date_time_forecast=date_time_forecast,
            date_time_max=date_time_max,
            model_settings=model_settings,
            model_id=model_id,
            write_db=write_db,
        )
        reporting.append(batch_report)

    batches_to_process = get_batches_to_reschedule(reporting)

if len(batches_to_process) != 0:
    msg = f"There are still the following batches left to reschedule: {batches_to_process}"
    logging.warning(msg)
    raise Exception(msg)

depvar = model_settings["depvar"]
logging.info(f"Finished predictions with depvar: {depvar}")

# optimize table final model_predictions_grid
