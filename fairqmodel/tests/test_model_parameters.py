from pytest import raises

from fairqmodel.model_parameters import get_variables, get_xgboost_param, load_json


def test_get_xgboost_param_output():
    """
    Checks output for correct format, type and keywords.
    """
    # arrange
    depvar = "no2"
    target_keys = {"max_depth", "eta", "objective", "nthread", "eval_metric"}

    # act
    res, n_rounds = get_xgboost_param(depvar)
    actual_keys = set(res.keys())

    # assert
    assert isinstance(res, dict)
    assert target_keys.issubset(actual_keys)
    assert isinstance(n_rounds, int)


def test_get_variables_input():
    """
    Checks input for format and depvar.
    """
    # arrange
    depvar = "im_not_a_depvar"
    lags = [3, 1, 4]

    # act/assert
    with raises(ValueError):
        get_variables(depvar, lags)


def test_get_variables_output():
    """
    Checks output for correct format, type and keywords.
    """
    # arrange
    depvar = "no2"
    lags = [3, 1, 4]

    # act
    res_all, res_metric, res_categoric = get_variables(depvar, lags)

    # assert
    assert isinstance(res_all, list)
    assert isinstance(res_metric, list)
    assert isinstance(res_metric, list)
    assert all(isinstance(v, str) for v in res_all + res_metric + res_categoric)
    assert len(res_all) == len(res_metric) + len(res_categoric)
    assert f"{depvar}_lag1" in res_metric  # wurde der Variablenname richtig gebaut


def test_get_xgboost_param_error():
    """
    Checks for correct depvar.
    """
    # arrange
    depvar = "pm42"

    # act/assert
    with raises(KeyError):
        get_xgboost_param(depvar)


def test_load_json_path():
    """
    Test that function checks for valid selector.
    """
    # arrange

    # valid parameters
    target_dir_v = "params"
    target_filename_v = "limit_values"

    # invalid parameters
    selector_iv = "not_a_selector"

    # select/assert
    with raises(KeyError, match="Used invalid key to query the file: not_a_selector"):
        load_json(target_dir_v, target_filename_v, selector_iv)


def test_load_json_output():
    """
    Test that function returns the correct type.
    """
    # arrange

    # valid paramters
    target_dir = "params"
    target_filename = "variables"
    selector = "no2_var"

    # select
    res = load_json(target_dir, target_filename, selector)

    # assert
    assert isinstance(res, dict)


def test_dev_mode():
    """
    Checks if DEV mode uses fewer variables than standard mode.
    """
    # arrange
    depvar = "no2"
    lags = [3, 1, 4]
    dev_on = True
    dev_off = False

    # act
    res_on_features, _, _ = get_variables(depvar, lags, dev=dev_on)
    res_off_features, _, _ = get_variables(depvar, lags, dev=dev_off)

    # assert
    assert len(res_on_features) < len(res_off_features)
