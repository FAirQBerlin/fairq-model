from typing import List, Optional, Tuple

from fairqmodel.model_parameters import get_features_first_stage


def assign_features_to_stage(use_two_stages: bool, feature_cols: List[str]) -> Tuple[List[str], Optional[List[str]]]:
    """Assigns the features  to the stages of the model.
    If a one-stage model is used, all features are assigned to the first stage.
    If a two-stage model is used, a specified subset is used for the fist stage
    and all features are used in the second stage.

    :param use_two_stages: bool, Specifies if one or two stages are used
    :param feature_cols: List[str], Containing all features that should be used

    :return: Tuple[List[str], Optional[List[str]]]
    """

    if not use_two_stages:
        features_stage_1 = feature_cols
        features_stage_2 = None
    else:
        features_allowed_in_first_stage = get_features_first_stage()
        features_stage_1 = [feat for feat in feature_cols if feat in features_allowed_in_first_stage]
        features_stage_2 = feature_cols

    return features_stage_1, features_stage_2
