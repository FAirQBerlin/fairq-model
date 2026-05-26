"""Unit tests for read_write_model_aux_functions module."""

import pytest

from fairqmodel.read_write_model_aux_functions import model_name_str


def test_model_name_str_spatial():
    """Check that 'spatial' returns the expected model name string."""
    result = model_name_str("spatial")
    assert result == "full_data_spatial"


def test_model_name_str_temporal():
    """Check that 'temporal' returns the expected model name string."""
    result = model_name_str("temporal")
    assert result == "full_data_temporal"


def test_model_name_str_all_uses_wildcard():
    """Check that 'all' returns a wildcard model name string."""
    result = model_name_str("all")
    assert result == "full_data_%"


def test_model_name_str_default_is_all():
    """Check that the default argument returns the wildcard model name."""
    result = model_name_str()
    assert result == "full_data_%"


def test_model_name_str_invalid_raises():
    """Check that an invalid target raises an AssertionError."""
    with pytest.raises(AssertionError):
        model_name_str("invalid_target")
