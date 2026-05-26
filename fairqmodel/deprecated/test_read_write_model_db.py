"""Integration tests for read_write_model_db using a real ClickHouse container.

These tests spin up a ClickHouse container, create the minimal required schema,
and verify that save_model_to_db / retrieve_model_from_db round-trip correctly.
"""

import datetime
from contextlib import contextmanager
from unittest.mock import patch

import clickhouse_connect
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from testcontainers.clickhouse import ClickHouseContainer

from fairqmodel.db_connect import ClickHouseWrapper
from fairqmodel.model_wrapper import ModelWrapper
from fairqmodel.read_write_model_db import retrieve_model_from_db, save_model_to_db

# ---------------------------------------------------------------------------
# Schema for the minimal model_description table (ReplacingMergeTree so that
# send_data_clickhouse's "replace" mode works)
# ---------------------------------------------------------------------------
CREATE_MODEL_DESCRIPTION = """
CREATE TABLE IF NOT EXISTS test.model_description
(
    model_id                      UInt32,
    date_time_training_execution  DateTime,
    pollutant                     String,
    model_name                    String,
    description                   String,
    model_object                  String,
    description_residuals         Nullable(String),
    model_object_residuals        Nullable(String)
)
ENGINE = ReplacingMergeTree()
ORDER BY model_id
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ch_container():
    """Provide a ClickHouse container for integration tests."""
    with ClickHouseContainer("clickhouse/clickhouse-server:latest") as ch:
        yield ch


@pytest.fixture(scope="module")
def raw_client(ch_container):
    """Provide a raw ClickHouse client with the minimal schema initialized."""
    client = clickhouse_connect.get_client(
        host=ch_container.get_container_host_ip(),
        port=int(ch_container.get_exposed_port(8123)),
        username=ch_container.username,
        password=ch_container.password,
        database=ch_container.dbname,
        secure=False,
        verify=False,
    )
    # Create schema + table
    client.command("CREATE DATABASE IF NOT EXISTS test")
    client.command(CREATE_MODEL_DESCRIPTION)
    yield client
    client.close()


@pytest.fixture
def wrapper(raw_client):
    """Return a fresh ClickHouseWrapper pointing at the container."""
    return ClickHouseWrapper(raw_client)


@contextmanager
def _patch_db(wrapper: ClickHouseWrapper):
    """Patch both db_connect_target and send_data_clickhouse to use the container wrapper.

    Uses the container wrapper instead of the real DB credentials.
    """

    @contextmanager
    def _fake_connect():
        yield wrapper

    def _fake_send(df, table_name, mode="replace", schema_name="test"):  # noqa: ARG001
        # Insert directly - bypass TLS / env-var lookups
        wrapper._client.insert_df(df=df, table=f"test.{table_name}")  # noqa: SLF001
        if mode == "replace":
            wrapper._client.command(f"OPTIMIZE TABLE test.{table_name} FINAL")  # noqa: SLF001
        return True

    with (
        patch("fairqmodel.read_write_model_db.db_connect_target", side_effect=_fake_connect),
        patch("fairqmodel.read_write_model_db.send_data_clickhouse", side_effect=_fake_send),
    ):
        yield


# ---------------------------------------------------------------------------
# Helper: train a tiny XGBoost model
# ---------------------------------------------------------------------------


def _tiny_booster() -> xgb.Booster:
    rng = np.random.default_rng(42)
    X = rng.random((50, 3))  # noqa: N806
    y = rng.random(50)
    dm = xgb.DMatrix(X, label=y, feature_names=["f1", "f2", "f3"])
    return xgb.train({"max_depth": 2, "n_estimators": 5}, dm, num_boost_round=5)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_save_and_retrieve_single_stage_model(wrapper):
    """Check that a single-stage model can be saved and retrieved with correct attributes."""
    booster = _tiny_booster()
    models = ModelWrapper(depvar="no2", model_1=booster)

    with _patch_db(wrapper):
        model_id = save_model_to_db(
            models=models,
            model_name="full_data_spatial",
            model_1_description="test model",
            model_2_description=None,
            execution_time=datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
        )

        retrieved = retrieve_model_from_db(model_id)

    assert isinstance(retrieved, ModelWrapper)
    assert retrieved.depvar == "no2"
    assert retrieved.model_1 is not None
    assert retrieved.model_2 is None
    assert retrieved.is_trained is True
    assert retrieved.use_two_stages is False


def test_save_and_retrieve_two_stage_model(wrapper):
    """Check that a two-stage model can be saved and retrieved with both stages intact."""
    booster_1 = _tiny_booster()
    booster_2 = _tiny_booster()
    models = ModelWrapper(depvar="pm25", model_1=booster_1, model_2=booster_2)

    with _patch_db(wrapper):
        model_id = save_model_to_db(
            models=models,
            model_name="full_data_temporal",
            model_1_description="stage 1",
            model_2_description="stage 2 residuals",
            execution_time=datetime.datetime(2024, 6, 1, 8, 0, 0, tzinfo=datetime.UTC),
        )

        retrieved = retrieve_model_from_db(model_id)

    assert retrieved.depvar == "pm25"
    assert retrieved.model_1 is not None
    assert retrieved.model_2 is not None
    assert retrieved.use_two_stages is True


def test_retrieved_model_produces_same_predictions(wrapper):
    """Check that a retrieved model produces identical predictions to the original."""
    booster = _tiny_booster()
    models = ModelWrapper(depvar="no2", model_1=booster)

    rng = np.random.default_rng(0)
    X_test = pd.DataFrame(rng.random((10, 3)), columns=["f1", "f2", "f3"])  # noqa: N806

    pred_before, _, _ = models.predict(X_test)

    with _patch_db(wrapper):
        model_id = save_model_to_db(
            models=models,
            model_name="full_data_spatial",
            model_1_description="round-trip test",
            model_2_description=None,
            execution_time=datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC),
        )
        retrieved = retrieve_model_from_db(model_id)

    pred_after, _, _ = retrieved.predict(X_test)
    np.testing.assert_array_almost_equal(pred_before, pred_after)


def test_save_increments_model_id(wrapper):
    """Check that each saved model receives an incremented model_id."""
    booster = _tiny_booster()
    models = ModelWrapper(depvar="no2", model_1=booster)

    with _patch_db(wrapper):
        id_1 = save_model_to_db(
            models=models,
            model_name="full_data_spatial",
            model_1_description="first",
            model_2_description=None,
            execution_time=datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC),
        )
        id_2 = save_model_to_db(
            models=models,
            model_name="full_data_spatial",
            model_1_description="second",
            model_2_description=None,
            execution_time=datetime.datetime(2024, 1, 2, tzinfo=datetime.UTC),
        )

    assert id_2 == id_1 + 1
