import logging
import sys
from logging.config import dictConfig

from logging_config.logger_config import get_logger_config

dictConfig(get_logger_config())


def get_command_args(argument):
    argument_val = [arg for arg in sys.argv if argument in arg]
    if argument_val == []:
        argument_val = None
    else:
        argument_val = argument_val[0].split("=")[1]

        if argument_val in ["True", "False"]:
            argument_val = bool(eval(argument_val))
        elif argument_val not in ["no2", "pm10", "pm25", "grid", "grid_sim"]:
            argument_val = int(argument_val)

    logging.info(f"Command line arg {argument} = {argument_val}")

    return argument_val
