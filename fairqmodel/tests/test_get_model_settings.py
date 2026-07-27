"""Unit tests for get_model_settings with mocked database access."""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import xgboost as xgb

from fairqmodel.model_wrapper import ModelWrapper
from fairqmodel.prediction_t_plus_k import get_model_settings


def _tiny_booster(feature_names: list[str]) -> xgb.Booster:
    """Create a minimal trained booster with given feature names."""
    rng = np.random.default_rng(42)
    n_samples = 20
    x_data = pd.DataFrame(rng.random((n_samples, len(feature_names))), columns=feature_names)
    y = rng.random(n_samples)
    dtrain = xgb.DMatrix(x_data, label=y, feature_names=feature_names)
    return xgb.train({"max_depth": 1, "verbosity": 0}, dtrain, num_boost_round=2)


def _make_mock_wrapper_single_stage() -> ModelWrapper:
    """Create a single-stage ModelWrapper with metadata."""
    feature_names = ["f1", "f2", "f3"]
    booster = _tiny_booster(feature_names)
    description = json.dumps(
        {
            "lags": "[1, 2]",
            "lags_avg": "[(1, 2)]",
            "training_period": ["2022-01-01", "2023-01-01"],
        }
    )
    return ModelWrapper(
        depvar="no2",
        model_1=booster,
        description=description,
        description_residuals=None,
    )


def _make_mock_wrapper_two_stage() -> ModelWrapper:
    """Create a two-stage ModelWrapper with metadata."""
    feature_names_1 = ["f1", "f2", "f3"]
    feature_names_2 = ["f2", "f3", "f4"]
    booster_1 = _tiny_booster(feature_names_1)
    booster_2 = _tiny_booster(feature_names_2)
    description = json.dumps(
        {
            "lags": "[1, 2]",
            "lags_avg": "[(1, 2)]",
            "training_period": ["2022-01-01", "2023-01-01"],
        }
    )
    description_residuals = json.dumps(
        {
            "lags": "[3]",
            "lags_avg": "[(2, 4)]",
            "training_period": ["2022-01-01", "2023-06-01"],
        }
    )
    return ModelWrapper(
        depvar="pm10",
        model_1=booster_1,
        model_2=booster_2,
        description=description,
        description_residuals=description_residuals,
    )


@patch("fairqmodel.prediction_t_plus_k.retrieve_model_from_db")
def test_get_model_settings_single_stage(mock_retrieve):
    """Test get_model_settings returns correct dict for a single-stage model."""
    # arrange
    model_id = 1027
    mock_retrieve.return_value = _make_mock_wrapper_single_stage()

    # act
    model_settings = get_model_settings(model_id)

    # assert
    assert isinstance(model_settings, dict)
    expected_keys = {
        "models",
        "feature_cols",
        "categorical_feature_cols",
        "metric_feature_cols",
        "lags",
        "lags_avg",
        "depvar",
        "max_training_date_model_1",
        "max_training_date_model_2",
    }
    assert set(model_settings.keys()) == expected_keys
    assert model_settings["depvar"] == "no2"
    assert model_settings["lags"] == [1, 2]
    assert model_settings["lags_avg"] == [(1, 2)]
    assert model_settings["max_training_date_model_1"] == "2023-01-01"
    assert model_settings["max_training_date_model_2"] is None
    mock_retrieve.assert_called_once_with(model_id)


@patch("fairqmodel.prediction_t_plus_k.retrieve_model_from_db")
def test_get_model_settings_two_stage(mock_retrieve):
    """Test get_model_settings returns correct dict for a two-stage model."""
    # arrange
    model_id = 2000
    mock_retrieve.return_value = _make_mock_wrapper_two_stage()

    # act
    model_settings = get_model_settings(model_id)

    # assert
    assert isinstance(model_settings, dict)
    assert model_settings["depvar"] == "pm10"
    assert model_settings["lags"] == [1, 2, 3]
    assert model_settings["lags_avg"] == [(1, 2), (2, 4)]
    assert model_settings["max_training_date_model_1"] == "2023-01-01"
    assert model_settings["max_training_date_model_2"] == "2023-06-01"
    # feature_cols should contain features from both stages (deduplicated)
    assert set(model_settings["feature_cols"]) == {"f1", "f2", "f3", "f4"}
    mock_retrieve.assert_called_once_with(model_id)


@patch("fairqmodel.prediction_t_plus_k.retrieve_model_from_db")
def test_get_model_settings_only_requires_model_id(mock_retrieve):
    """Test that get_model_settings signature only requires model_id."""
    # arrange
    model_id = 999
    mock_retrieve.return_value = _make_mock_wrapper_single_stage()

    # act - should work with just model_id
    model_settings = get_model_settings(model_id)

    # assert
    assert isinstance(model_settings, dict)
    assert isinstance(model_settings["models"], ModelWrapper)
