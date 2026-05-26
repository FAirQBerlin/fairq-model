"""Process grid prediction batches for all depvars and write results to the DB."""

import sys

from loguru import logger

import fairqmodel as fqm
from fairqmodel.command_line_args import get_command_args
from fairqmodel.process_batch import process_batch

logger.remove()
logger.add(sys.stdout, level="INFO")

logger.info(f"Using fairqmodel in version: {fqm.__version__}")


model_type = get_command_args("model_type") or "grid"
write_db = get_command_args("write_db") or False
msg = get_command_args("msg")

logger.info(f"Starting with msg = {msg}")

if not msg:
    raise ValueError('msg argument is required. Example: msg=\'{"batch_id": 1, "depvar": "no2"}\'')


process_batch(msg, write_db=bool(write_db), mode=str(model_type))
