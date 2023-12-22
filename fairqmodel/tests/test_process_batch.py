import os

import pytest

from fairqmodel.process_batch import process_batch


def test_process_batch_grid():
    if "INWT-L" not in os.uname()[1]:  # currently only local testing
        pytest.skip("no yet supported on jenkins")

    # arrange
    msg = "{'batch_id': '1', 'depvar': 'no2'}"
    exp = {"batch": 1, "finished": True}

    # act
    res = process_batch(msg, write_db=False, mode="grid")

    # assert
    assert res == exp


def test_process_batch_grid_sim():
    if "INWT-L" not in os.uname()[1]:  # currently only local testing
        pytest.skip("no yet supported on jenkins")

    # arrange
    msg = "{'batch_id': '1', 'depvar': 'no2'}"
    exp = {"batch": 1, "finished": True}

    # act
    res = process_batch(msg, write_db=False, mode="grid_sim")

    # assert
    assert res == exp
