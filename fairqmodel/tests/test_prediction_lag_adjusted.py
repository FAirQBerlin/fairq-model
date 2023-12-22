from fairqmodel.prediction_t_plus_k import make_lag_adjusted_prediction
from fairqmodel.tests.helpers import arrange_for_lag_pred_tests


def test_make_lag_adjusted_prediction():
    dat, model_settings, _, _ = arrange_for_lag_pred_tests()
    fold = {"test": dat.loc[dat.pm25.isnull(), :], "ts_fold_max_train_date": "dummy_entry"}

    res_df, _ = make_lag_adjusted_prediction(
        fold=fold,
        models=model_settings["models"],
        feature_cols=model_settings["feature_cols"],
        depvar=model_settings["depvar"],
        lags_actual=model_settings["lags"],
        lags_avg=model_settings["lags_avg"],
        calc_metrics=False,
        testing=True,
    )

    assert res_df.pred.tolist() == [
        12.319515228271484,
        13.60685920715332,
        14.026861190795898,
        13.890061378479004,
        14.066526412963867,
        14.066526412963867,
        14.066526412963867,
        14.066526412963867,
        14.066526412963867,
        14.066526412963867,
        14.066526412963867,
        14.066526412963867,
        14.066526412963867,
        14.066526412963867,
        14.066526412963867,
    ]

    assert res_df.pm25_lag1[1:].tolist() == res_df.pred[:-1].tolist()
    assert res_df.pm25_lag2[2:].tolist() == res_df.pred[:-2].tolist()
    assert res_df.pm25_lag3[3:].tolist() == res_df.pred[:-3].tolist()


# __file__ = "fairqmodel/tests/test_prediction_lag_adjusted.py"
