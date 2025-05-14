import logging
from logging.config import dictConfig

from fairqmodel.command_line_args import get_command_args
from fairqmodel.produce_events import produce_json_events
from logging_config.logger_config import get_logger_config

dictConfig(get_logger_config())

model_type = get_command_args("model_type") or "grid"

produce_json_events(str(model_type))

logging.info("Successfully produced events}.")
