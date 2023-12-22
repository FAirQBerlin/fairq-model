import copy
import json
from typing import List, Optional, Tuple

import pandas as pd

from fairqmodel.model_wrapper import ModelWrapper


def create_model_description(
    models: ModelWrapper,
    dat: pd.DataFrame,
    lags_actual: List[int],
    lags_avg: List[int],
) -> Tuple[str, Optional[str]]:
    """Creates a description of every model in the model wrapper in a string format.
    If only one stage is used, the second description is None.

    :param models: ModelWrapper, Wrapping object containing one or two models
    :param dat: pd.DataFrame, Data used for training
    :param lags_actual: list, Lags used in training
    :param lags_avg: list, Lags for average feature

    :return: Model description string(s). In case of a one-stage model, the second description is None.
    """
    if models.use_two_stages:
        # If two models are used, the first one doesn't use lags
        model_1_description = create_one_description(models.xgb_param_1, dat, lags_actual=[], lags_avg=[])
        model_2_description = create_one_description(models.xgb_param_2, dat, lags_actual, lags_avg)
    else:
        model_1_description = create_one_description(models.xgb_param_1, dat, lags_actual, lags_avg)
        # If just one model is used, there is no description for a second model
        model_2_description = None

    return model_1_description, model_2_description


def create_one_description(xgb_param: dict, dat: pd.DataFrame, lags_actual: List[int], lags_avg: List[int]) -> str:
    """Creates the string description from a given set of parameters.

    :param xgb_param: dict, Parameters of the model for which the description is made
    :param dat: pd.DataFrame, Contains the data on which the model was trained.
    :param lags_actual: list, Lags used in training
    :param lags_avg: list, Lags for average feature

    :return: String description of the model.
    """
    assert xgb_param is not None, "Tried to create a model description for model unset model parameters."
    model_description = copy.deepcopy(xgb_param)
    train_date_min = dat.date_time.min().strftime("%Y-%m-%d %H:%M:%S")
    train_date_max = dat.date_time.max().strftime("%Y-%m-%d %H:%M:%S")
    model_description["training_period"] = [train_date_min, train_date_max]
    model_description["lags"] = json.dumps(lags_actual)  # 'lags' is used as key, to be consitent with the old models
    model_description["lags_avg"] = json.dumps(lags_avg)

    model_description_str = json.dumps(dict(sorted(model_description.items())))

    return model_description_str
