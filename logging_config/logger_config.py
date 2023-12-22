"""Helper function to load the logger configuration from json."""
import json
from pathlib import Path


def get_logger_config() -> dict:
    """Read in the logger_config.json and set loglevel."""
    fpath = Path(__file__).parent / "logger_config.json"
    with open(fpath, "r") as conf_con:
        return json.load(conf_con)
