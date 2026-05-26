"""Produce grid prediction batch events for downstream processing."""

import sys

from loguru import logger

from fairqmodel.command_line_args import get_command_args
from fairqmodel.produce_events import produce_json_events

logger.remove()
logger.add(sys.stdout, level="INFO")

model_type = get_command_args("model_type") or "grid"

produce_json_events(str(model_type))

logger.info("Successfully produced events.")
