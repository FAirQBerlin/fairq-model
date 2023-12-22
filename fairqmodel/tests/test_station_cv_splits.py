import pandas as pd
from pytest import raises

from fairqmodel.station_cv_splits import get_station_cv_folds


def get_station_cv_folds_input():
    """
    Test that function fails if DataFrame has no column station_id.
    """
    # arrange
    dummy_dat = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

    # act / assert
    with raises(AttributeError):
        _ = get_station_cv_folds(dummy_dat)


def get_station_cv_folds_output():
    """
    Checks the returned format, size and content.
    """
    # arrange
    station_ids = [1, 1, 3, 4]
    dummy_dat = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": [4, 5, 6, 7], "station_id": station_ids})
    target_keys = {"station_id", "train", "test"}

    train_idx = dummy_dat.loc[dummy_dat.station_id != station_ids[0], :].index.values.tolist()
    test_idx = dummy_dat.loc[dummy_dat.station_id == station_ids[0], :].index.values.tolist()

    # act
    station_cv_folds = get_station_cv_folds(dummy_dat)
    res_keys = set(station_cv_folds[0].keys())

    res_train_idx = station_cv_folds[0]["train"]
    res_test_idx = station_cv_folds[0]["test"]

    # assert
    assert isinstance(station_cv_folds, list)
    assert isinstance(station_cv_folds[0], dict)
    assert len(station_cv_folds) == len(dummy_dat.station_id.unique())
    assert res_keys == target_keys
    assert set(station_cv_folds[0].keys()) == set(station_cv_folds[1].keys()) == set(station_cv_folds[2].keys())

    # check if test set only contains entries from selected station
    assert sorted(test_idx) == sorted(res_test_idx)
    # check if train set only contains entries from remaining stations
    assert sorted(train_idx) == sorted(res_train_idx)
