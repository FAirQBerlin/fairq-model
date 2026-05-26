"""Parse command-line arguments for model scripts."""

import sys

from loguru import logger


def get_command_args(argument):
    """Parse and return the value of the specified command-line argument."""
    argument_val = [arg for arg in sys.argv if argument in arg]

    if argument_val == []:
        argument_val = None
    else:
        argument_val = argument_val[0].split("=")[1]

        if argument == "msg":
            return argument_val

        if argument_val in ["True", "False"]:
            argument_val = bool(eval(argument_val))
        elif argument_val not in ["no2", "pm10", "pm25", "grid", "grid_sim"]:
            argument_val = int(argument_val)

    logger.info(f"Command line arg {argument} = {argument_val}")

    return argument_val
