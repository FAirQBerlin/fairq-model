"""Auxiliary functions for serializing and deserializing XGBoost model objects."""

import json
from pathlib import Path

import numpy as np
import xgboost as xgb


def model_from_str(model_object_str) -> xgb.Booster:
    """Convert a model from string representation to xgb format.

    :param model_object_str: str, Model in string format

    :return: xgb.Booster, Model in xgb format
    """
    random_suffix = np.random.default_rng().integers(0, 1_000_000)

    # Convert str to dict
    model = json.loads(model_object_str)

    # Save dict locally
    with Path(f"temp_model_backward_{random_suffix}.json").open("w") as f:
        json.dump(model, f)

    # Load the model
    model = xgb.Booster()
    model.load_model(f"temp_model_backward_{random_suffix}.json")

    # Delete local file
    Path(f"temp_model_backward_{random_suffix}.json").unlink()

    return model


def model_as_str(model: xgb.Booster) -> str:
    """Convert an xgb model to a string representation.

    :param model: xgb.Booster, Model to be converted

    :return: str, String representation of the model
    """
    random_suffix = np.random.default_rng().integers(0, 1_000_000)

    # Save model locally
    model.save_model(f"temp_model_forward_{random_suffix}.json")

    # Load local file
    with Path(f"temp_model_forward_{random_suffix}.json").open() as f:
        model_object_dict = json.load(f)

    # Delete local file
    Path(f"temp_model_forward_{random_suffix}.json").unlink()

    # Convert model_object to str
    return json.dumps(model_object_dict)
