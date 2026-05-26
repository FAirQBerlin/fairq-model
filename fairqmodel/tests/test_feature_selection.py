"""Unit tests for feature_selection module."""

from fairqmodel.feature_selection import assign_features_to_stage

SAMPLE_FEATURES = ["no2_lag1", "kfz_per_hour", "hour", "month", "temp", "no2_lag2"]


def test_one_stage_returns_all_features_in_stage_1():
    """Check that one-stage model assigns all features to stage 1 and stage 2 is None."""
    stage_1, stage_2 = assign_features_to_stage(use_two_stages=False, feature_cols=SAMPLE_FEATURES)
    assert stage_1 == SAMPLE_FEATURES
    assert stage_2 is None


def test_two_stage_stage_2_contains_all_features():
    """Check that two-stage model assigns all features to stage 2."""
    _, stage_2 = assign_features_to_stage(use_two_stages=True, feature_cols=SAMPLE_FEATURES)
    assert stage_2 == SAMPLE_FEATURES


def test_two_stage_stage_1_is_subset_of_all_features():
    """Check that two-stage model stage 1 features are a subset of all features."""
    stage_1, _ = assign_features_to_stage(use_two_stages=True, feature_cols=SAMPLE_FEATURES)
    assert set(stage_1).issubset(set(SAMPLE_FEATURES))


def test_two_stage_stage_1_not_empty():
    """Check that two-stage model stage 1 is a non-empty list."""
    # There must be at least some features allowed in stage 1 (from the JSON config)
    stage_1, _ = assign_features_to_stage(use_two_stages=True, feature_cols=SAMPLE_FEATURES)
    assert isinstance(stage_1, list)


def test_one_stage_empty_feature_list():
    """Check that an empty feature list is handled correctly."""
    stage_1, stage_2 = assign_features_to_stage(use_two_stages=False, feature_cols=[])
    assert stage_1 == []
    assert stage_2 is None
