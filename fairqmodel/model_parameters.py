import json
import os
from typing import List, Optional, Tuple


def get_xgboost_param(depvar: str, use_two_stages: bool = False, stage: int = 1, dev: bool = False) -> tuple[dict, int]:
    """Loads the optimal set of xgboost parameters from .json file

    :param depvar: str, Selected dependent variable
    :param use_two_stages: bool, Specifies if one or two models are used.
    :param stage: int, Specifies if parameters for first or the second model should be selected
    :param dev: bool,  Specifies if simple dummy parameters are used (for developing)

    :return: dict, optimized xgboost parameters
    :return: int, number of training rounds
    """
    stage_info = "solo" if not use_two_stages else f"stage_{stage}"
    selector = f"{depvar}_{stage_info}" if not dev else "dev"
    params = load_json(target_dir="params", target_filename="xgboost_params", selector=selector)
    n_rounds = params.pop("n_rounds")
    return params, n_rounds


def remove_weighted_average_vars(metric_variables: list[str]) -> list[str]:
    """Removes variables with 'wavg' (weighted average) in the name.
    This is necessary for the model at the stations, i.e. with lags,
    as the station model cannot use its own weighted averages (circularity)

    :param metric_variables: List[str], names of metric variables

    :return: list[str] names of metric variables.
    """
    return [var for var in metric_variables if "wavg" not in var]


def get_variables(
    depvar: str, lags_actual: List[int], lags_avg: List[int] = [], dev: bool = False
) -> tuple[list[str], list[str], list[str]]:
    """Loads the list of available variable names from .json file

    :param depvar: str, selected dependent variable
    :param lags_actual: List[int], lag values for each data point
    :param lags_avg: List[int], If not empty, lags to build an average feature
    :param dev: bool, If set, use small subset of variables for developing

    :return: list[list[str], list[str], list[str]], nested list where inner lists are:
             names of all variables, names of metric variables, names of categoric variables.
    """

    if depvar not in ["pm10", "pm25", "no2"]:
        raise ValueError(f"Selected invalid dependent variable: {depvar}")

    selector = "dev_var" if dev else f"{depvar}_var"

    temp = load_json(target_dir="params", target_filename="variables", selector=selector)

    metric_variables = temp["metric"]
    categoric_variables = temp["categoric"]

    if (len(lags_actual) > 0) | (len(lags_avg) > 0):  # at stations with lags
        metric_variables = remove_weighted_average_vars(metric_variables)

    lags_temp = [f"{depvar}_lag{lag}" for lag in lags_actual]

    metric_variables.extend(lags_temp)

    if len(lags_avg) > 0:
        avg_feature = f"lag_avg_{sorted(lags_avg)}".replace("[", "(").replace("]", ")")
        metric_variables.append(avg_feature)

    all_features = categoric_variables + metric_variables
    return all_features, metric_variables, categoric_variables


def load_json(target_dir: str, target_filename: str, selector: str) -> dict:
    """Loads the dictionary stored in a .json file.
    Here, used to either load model parameters or variable names.

    :param target_dir: str, name of the parent directory of the .json file
    :param target_filename: str, name of the .json file
    :param selector: str, key word to query the json

    :return: dict, selected content from the .json
    """
    temp = json.load(open(os.path.join(os.path.dirname(__file__), target_dir, f"{target_filename}.json")))
    try:
        return temp[selector]
    except KeyError:
        raise KeyError(f"Used invalid key to query the file: {selector}.")


def get_pollution_limits() -> dict:
    """Loads the allowed limit values per pollutant.

    :return: dict
    """
    limit_values = load_json(target_dir="params", target_filename="limit_values", selector="limit_values")

    return limit_values


def get_tweak_values() -> dict:
    """Loads the optimized tweak values per pollutant.

    :return: dict
    """
    tweak_values = load_json(target_dir="params", target_filename="limit_values", selector="tweak_values")

    return tweak_values


def get_train_date_min(depvar: str) -> dict:
    """get the train date min for one depvar

    :return: dict
    """
    if depvar not in ["pm10", "pm25", "no2"]:
        raise ValueError(f"Selected invalid dependent variable: {depvar}")

    date_min = load_json(target_dir="params", target_filename="train_date_min", selector="date_min")

    return date_min[depvar]


def get_lags(
    use_lags: bool = True, selected_lags: Optional[List[int]] = None, lags_avg: Optional[List[int]] = None
) -> Tuple[List[int], List[int]]:
    """Provides lag values. Per default returns either the optimized lags or an empty list.
    It is however possible to select other lags and lags to build an average feature.

    :param use_lags: bool, Specifies if any lags are used.
    :param selected_lags: Optional[List[int]], If provided, list of lags to use instead of the default values.
    :param lags_avg: Optional[List[int]], If provided, list of lags to build an average feature.

    :return: List with actual lags
    :return: List with lag values for an average feature

    """
    if use_lags:
        lags_actual = [24, 48] if selected_lags is None else selected_lags
        lags_avg = [1, 2, 3, 4, 5] if lags_avg is None else lags_avg
    else:
        lags_actual = []
        lags_avg = []

    # Remove potential duplicates and sort values
    lags_actual = sorted(list(set(lags_actual)))
    lags_avg = sorted(list(set(lags_avg)))

    return lags_actual, lags_avg


def get_lag_options(use_lags: bool, lags_avg: List[int] = []) -> Tuple[List[List[int]], List[int]]:
    """Provides access to lag combinations for the HPO.
    Since the optimization can only suggest a single value,
    the suggested int is used as an index to select one of
    the below lag combinations.
    If use_lags is True, the lags_avg are used as in the parameters.
    Otherwise the lags_avg is overwritten by an empty list.

    :param use_lags: bool, Specifies if any lags are used
    :param lags_avg: List[int], Lags to use for an average feature

    :return: List[List[int]]

    """
    lag_options: List[List[int]] = []

    if use_lags:
        lag_options = [
            [24, 48],
            get_lags()[0],  # this is the default setting, don't remove this as the first entry
            [],
            [3, 6, 8, 24],
            [3, 4, 6, 8, 24, 48],
            [3, 4, 6, 8, 24],
            [3, 6, 8, 24, 48],
            [4, 6, 8, 24, 48],
            [4, 6, 8, 24, 48, 72, 24 * 7],
            [4, 8, 24, 48],
            [4, 8, 24],
        ]
    else:
        lags_avg = []

    return lag_options, lags_avg


def get_features_first_stage() -> List[str]:
    """Loads the variables that are generally permitted in the first part of a two-stage model.
    Which of these are actually used, depends on the pollutant.

    :return: List[str]

    """
    variables = load_json(target_dir="params", target_filename="variables", selector="first_model_var")["var_names"]
    return variables
