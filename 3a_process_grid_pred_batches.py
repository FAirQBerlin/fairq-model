import logging
from logging.config import dictConfig

import fairqmodel as fqm
from fairqmodel.command_line_args import get_command_args
from fairqmodel.process_batch import process_batch
from logging_config.logger_config import get_logger_config

dictConfig(get_logger_config())

logging.info("Using fairqmodel in version: {}".format(fqm.__version__))


model_type = get_command_args("model_type") or "grid"
write_db = get_command_args("write_db") or False
msg = get_command_args("msg")

logging.info(f"Starting with msg = {msg}")

if not msg:
    # for testing you can uncomment the following line:
    # msg = {"batch_id": 1, "depvar": "no2"}
    raise ValueError('msg argument is required. Example: msg=\'{"batch_id": 1, "depvar": "no2"}\'')


process_batch(msg, write_db=bool(write_db), mode=str(model_type))
