"""Integration tests for grid prediction batch processing."""

import pytest
from loguru import logger

import fairqmodel as fqm
from fairqmodel.command_line_args import get_command_args
from fairqmodel.process_batch import process_batch


@pytest.mark.tests_on_real_clickhouse
def test_process_grid_pred_batches():
    """Test the process_batch function for grid prediction batches."""
    # arrange
    expected = {"batch": 1, "finished": True}
    msg = '{"batch_id": 1, "depvar": "no2"}'

    # act
    logger.info(f"Using fairqmodel in version: {fqm.__version__}")

    model_type = get_command_args("model_type") or "grid"
    write_db = get_command_args("write_db") or False

    logger.info(f"Starting with msg = {msg}")

    if not msg:
        raise ValueError('msg argument is required. Example: msg=\'{"batch_id": 1, "depvar": "no2"}\'')

    res = process_batch(msg, write_db=bool(write_db), mode=str(model_type))

    # assert the result is as expected
    assert isinstance(res, dict), "Result should be a dictionary"
    assert "batch" in res, "Result should contain 'batch' key"
    assert res["batch"] == expected["batch"]
    assert "finished" in res, "Result should contain 'finished' key"
    assert res["finished"] == expected["finished"]
