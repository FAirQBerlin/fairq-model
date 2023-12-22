import logging
import signal
from logging.config import dictConfig

from confluent_kafka import Consumer

from fairqmodel.command_line_args import get_command_args
from fairqmodel.db_connect import mode
from fairqmodel.process_batch import process_batch
from logging_config.logger_config import get_logger_config

dictConfig(get_logger_config())


def sigterm_handler(signal, frame):
    global running
    logging.info("Shutting down after next iteration")
    running = False


signal.signal(signal.SIGTERM, sigterm_handler)

model_type = get_command_args("model_type") or "grid"
topic = f"fairq-model-pred-{model_type.replace('_', '-')}"
broker = f"kafka.{mode().lower()}.inwt.de:443"
write_db = get_command_args("write_db") or False

c = Consumer(
    {
        "bootstrap.servers": broker,
        "group.id": f"{topic}-{mode()}",
        "enable.auto.commit": False,
        "security.protocol": "ssl",
        "enable.ssl.certificate.verification": "false",
        "auto.offset.reset": "earliest",
    }
)
logging.info("Kafka Consumer has been initiated...")

c.subscribe([topic])

logging.info(f"Partition: {c.assignment()}")

running = True

while running:
    msg = c.poll(1.0)
    logging.info("Polling")
    logging.info(f"Partition: {c.assignment()}")

    if msg is None:
        continue
    if msg.error():
        print("Consumer error: {}".format(msg.error()))
        continue

    logging.info("Partition {} - Received message: {}".format(msg.partition(), msg.value().decode("utf-8")))
    process_batch(msg.value().decode("utf-8"), write_db=write_db, mode=model_type)

    c.commit(asynchronous=False)

c.close()
