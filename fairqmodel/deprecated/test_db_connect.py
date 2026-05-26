"""Integration tests for db_connect module using a real ClickHouse container.

All tests are marked @pytest.mark.tests_on_real_clickhouse and are excluded from the default pytest run.
Run them explicitly with: pytest -m tests_on_real_clickhouse
"""

import clickhouse_connect
import pandas as pd
import pytest
from testcontainers.clickhouse import ClickHouseContainer

from fairqmodel.db_connect import ClickHouseWrapper


@pytest.fixture(scope="module")
def clickhouse_container():
    """Provide a ClickHouse container for integration tests."""
    with ClickHouseContainer("clickhouse/clickhouse-server:latest") as ch:
        yield ch


@pytest.fixture(scope="module")
def ch_client(clickhouse_container):
    """Provide a ClickHouseWrapper client connected to the test container."""
    client = clickhouse_connect.get_client(
        host=clickhouse_container.get_container_host_ip(),
        port=int(clickhouse_container.get_exposed_port(8123)),
        username=clickhouse_container.username,
        password=clickhouse_container.password,
        database=clickhouse_container.dbname,
        secure=False,
        verify=False,
    )
    yield ClickHouseWrapper(client)
    client.close()


def test_clickhouse_wrapper_execute(ch_client):
    """Check that execute returns a non-None result."""
    result = ch_client.execute("SELECT 1")
    assert result is not None


def test_clickhouse_wrapper_query_dataframe(ch_client):
    """Check that query_dataframe returns a non-empty DataFrame with expected columns."""
    df = ch_client.query_dataframe("SELECT 1 AS value")
    assert not df.empty
    assert "value" in df.columns
    assert df["value"].iloc[0] == 1


def test_clickhouse_wrapper_insert_and_query(ch_client):
    """Check that insert_dataframe inserts rows that can be queried back."""
    ch_client.execute("CREATE TABLE IF NOT EXISTS test_table (id UInt32, name String) ENGINE = MergeTree() ORDER BY id")
    df = pd.DataFrame({"id": [1, 2], "name": ["foo", "bar"]})
    ch_client.insert_dataframe("INSERT INTO test_table VALUES", df)

    result = ch_client.query_dataframe("SELECT * FROM test_table ORDER BY id")
    assert len(result) == 2  # noqa: PLR2004
    assert result["name"].tolist() == ["foo", "bar"]

    ch_client.execute("DROP TABLE IF EXISTS test_table")


def test_clickhouse_wrapper_context_manager(clickhouse_container):
    """Check that the context manager properly opens and closes the connection."""
    client = clickhouse_connect.get_client(
        host=clickhouse_container.get_container_host_ip(),
        port=int(clickhouse_container.get_exposed_port(8123)),
        username=clickhouse_container.username,
        password=clickhouse_container.password,
        database=clickhouse_container.dbname,
        secure=False,
        verify=False,
    )
    with ClickHouseWrapper(client) as wrapper:
        df = wrapper.query_dataframe("SELECT 42 AS answer")
    assert df["answer"].iloc[0] == 42  # noqa: PLR2004
