import json
import os

import numpy as np
import xgboost as xgb


def model_name_str(target: str = "all") -> str:
    """Constructs the full model name for the selected type of model,
    e.g. 'full_data_spatial', 'full_data_temporal'.
    This function is used to select/show available models from the DB.

    :param target: str, Type of models to select, must be one of "spatial", "temporal", "all"
    :return: str, Model name by which the models are selected.
    """
    assert target in ["spatial", "temporal", "all"]

    # The 'model_name' is used inside an SQL query
    # If "all" is selected, the string-wildcard '%' is used as target
    name = "full_data"
    temp_target = "%" if target == "all" else target
    model_name = f"{name}_{temp_target}"

    return model_name


def model_from_str(model_object_str) -> xgb.Booster:
    """Converts a model to xgb format from string.

    :param model_object_str: str, Model in string format

    :return: xgb.Booster, Model in xgb format
    """

    random_suffix = np.random.randint(0, 1000000)

    # Convert str to dict
    model = json.loads(model_object_str)

    # Save dict locally
    with open(f"temp_model_backward_{random_suffix}.json", "w") as f:
        json.dump(model, f)

    # Load the model
    model = xgb.Booster()
    model.load_model(f"temp_model_backward_{random_suffix}.json")

    # Delete local file
    os.remove(f"temp_model_backward_{random_suffix}.json")

    return model


def model_as_str(model: xgb.Booster) -> str:
    """Converts xgb model to a string.

    :param model: xgb.Booster, Model to be converted

    :return: str, String representation of the model
    """

    random_suffix = np.random.randint(0, 1000000)

    # Save model locally
    model.save_model(f"temp_model_forward_{random_suffix}.json")

    # Load local file
    with open(f"temp_model_forward_{random_suffix}.json", "r") as f:
        model_object_dict = json.load(f)

    # Delete local file
    os.remove(f"temp_model_forward_{random_suffix}.json")

    # Convert model_object to str
    model_object_str = json.dumps(model_object_dict)

    return model_object_str
