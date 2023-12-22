import logging
from logging.config import dictConfig

from confluent_kafka import Producer

from fairqmodel.command_line_args import get_command_args
from fairqmodel.db_connect import mode
from fairqmodel.kafka import produce_events
from logging_config.logger_config import get_logger_config

dictConfig(get_logger_config())

model_type = get_command_args("model_type") or "grid"
topic = f"fairq-model-pred-{model_type.replace('_', '-')}"
broker = f"kafka.{mode().lower()}.inwt.de:443"

p = Producer({"bootstrap.servers": broker, "security.protocol": "ssl", "enable.ssl.certificate.verification": "false"})
logging.info("Kafka Producer has been initiated...")

produce_events(topic, p, mode=model_type)

logging.info("Successfully produced events.")
