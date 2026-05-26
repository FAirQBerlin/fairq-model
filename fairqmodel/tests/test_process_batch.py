"""Integration tests for process_batch module."""

import pytest

from fairqmodel.process_batch import process_batch


@pytest.mark.tests_on_real_clickhouse
def test_process_batch_grid():
    """Check that a grid batch is processed successfully and returns the expected report."""
    # arrange
    msg = "{'batch_id': '1', 'depvar': 'no2'}"
    exp = {"batch": 1, "finished": True}

    # act
    res = process_batch(msg, write_db=False, mode="grid")

    # assert
    assert res == exp


@pytest.mark.tests_on_real_clickhouse
def test_process_batch_grid_sim():
    """Check that a grid_sim batch is processed successfully and returns the expected report."""
    # arrange
    msg = "{'batch_id': '1', 'depvar': 'no2'}"
    exp = {"batch": 1, "finished": True}

    # act
    res = process_batch(msg, write_db=False, mode="grid_sim")

    # assert
    assert res == exp
