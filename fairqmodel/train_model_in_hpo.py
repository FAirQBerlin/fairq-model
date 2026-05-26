"""Train XGBoost models for hyper-parameter optimization with early stopping."""

import pandas as pd
import xgboost as xgb
from loguru import logger

FIRST_STAGE = 1
SECOND_STAGE = 2


def train_hpo(
    dat: pd.DataFrame,
    dat_early_stopping: pd.DataFrame,
    depvar: str,
    feature_cols_1: list[str],
    feature_cols_2: list[str] | None,
    xgb_param: dict,
    n_rounds: int,
    early_stopping_rounds: int,
    early_stopping_threshold: float,
    first_stage_model: xgb.Booster | None = None,
) -> tuple[xgb.Booster, xgb.Booster | None]:
    """Train a model with given parameters for hyper parameter optimization.

    This function is used within the HPO.
    Only one stage can be optimized at a time.

    :param dat: pd.DataFrame, Contains the training data
    :param dat_early_stopping: pd.DataFrame, Contains hold-out-data to allow for early stopping
    :param depvar: str, Name of the dependent variable
    :param feature_cols_1: List[str], Containing variable names used in first stage
    :param feature_cols_2: Optional[List[str]], If two stages are used, containing variable names used in second stage
    :param xgb_param: dict, Model parameters suggested by the HPO
    :param n_rounds: int, number of training rounds suggested by the HPO
    :param early_stopping_rounds: int, If the model doesn't improve within this number of rounds,
                                       the training is terminated
    :param early_stopping_threshold: int, Specifies the minimal improvement that has to be
                                          achieved every <early_stopping_rounds> to continue training
    :param first_stage_model: Optional[xgb.booster], The model to use as a first stage, when optimizing
                                                     the second stage

    :return: Tuple[xgb.Booster, Optional[xgb.Booster]]
    """
    logger.info("Started model training")
    # Infer for which stage the given parameters are used
    stage_to_optimize = 1 if first_stage_model is None else 2

    # Define EarlyStopping callback to enable customized early stopping (i.e. passing a threshold value)
    callbacks = [
        xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds,
            metric_name="rmse",
            min_delta=early_stopping_threshold,
            save_best=False,
            maximize=False,
            data_name="val_rmse",
        )
    ]

    dmatrix_1 = xgb.DMatrix(dat.loc[:, feature_cols_1], label=dat[depvar], enable_categorical=True)

    # Case: A single model is optimized (either solo model or first of two-stage-model)
    if stage_to_optimize == FIRST_STAGE:
        model_1 = train_first_stage(
            depvar,
            xgb_param,
            dmatrix_1,
            feature_cols_1,
            n_rounds,
            early_stopping_rounds,
            dat_early_stopping,
            callbacks,
        )

    # Case: The second model of two-stage-model is optimized
    elif stage_to_optimize == SECOND_STAGE:
        assert first_stage_model is not None
        assert feature_cols_2 is not None
        model_2 = train_second_stage(
            depvar,
            xgb_param,
            dmatrix_1,
            dat,
            feature_cols_1,
            feature_cols_2,
            n_rounds,
            early_stopping_rounds,
            dat_early_stopping,
            callbacks,
            first_stage_model,
        )
    else:
        logger.warning(f"'stage_to_optimize' cannot be set to {stage_to_optimize}")

    first_stage = model_1 if stage_to_optimize == FIRST_STAGE else first_stage_model
    second_stage = model_2 if stage_to_optimize == SECOND_STAGE else None

    assert first_stage is not None

    logger.info("Finished model training")
    return first_stage, second_stage


def train_first_stage(
    depvar: str,
    xgb_param: dict,
    dmatrix_1: xgb.DMatrix,
    feature_cols_1: list[str],
    n_rounds: int,
    early_stopping_rounds: int,
    dat_early_stopping: pd.DataFrame,
    callbacks: list[xgb.callback.EarlyStopping],
) -> xgb.Booster:
    """Train the first stage model with early stopping."""
    dmatrix_evals_1 = [
        (
            xgb.DMatrix(
                dat_early_stopping.loc[:, feature_cols_1],
                label=dat_early_stopping[depvar],
                enable_categorical=True,
            ),
            "val_rmse",
        )
    ]
    return xgb.train(
        params=xgb_param,
        dtrain=dmatrix_1,
        evals=dmatrix_evals_1,
        num_boost_round=n_rounds,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
        callbacks=callbacks,
    )


def train_second_stage(
    depvar: str,
    xgb_param: dict,
    dmatrix_1: xgb.DMatrix,
    dat: pd.DataFrame,
    feature_cols_1: list[str],
    feature_cols_2: list[str],
    n_rounds: int,
    early_stopping_rounds: int,
    dat_early_stopping: pd.DataFrame,
    callbacks: list[xgb.callback.EarlyStopping],
    first_stage_model: xgb.Booster,
) -> xgb.Booster:
    """Train the second stage model on the residuals of the first stage."""
    # Create predictions from first stage
    prediction = first_stage_model.predict(dmatrix_1)
    residual = dat[depvar] - prediction
    dmatrix_2 = xgb.DMatrix(dat.loc[:, feature_cols_2], label=residual, enable_categorical=True)

    # Predict labels for eval set
    eval_dmatrix = xgb.DMatrix(
        dat_early_stopping.loc[:, feature_cols_1],
        label=dat_early_stopping[depvar],
        enable_categorical=True,
    )
    eval_prediction = first_stage_model.predict(eval_dmatrix)
    eval_residual = dat_early_stopping[depvar] - eval_prediction
    dmatrix_evals_2 = [
        (
            xgb.DMatrix(
                dat_early_stopping.loc[:, feature_cols_2],
                label=eval_residual,
                enable_categorical=True,
            ),
            "val_rmse",
        )
    ]

    # Train second model
    return xgb.train(
        params=xgb_param,
        dtrain=dmatrix_2,
        evals=dmatrix_evals_2,
        num_boost_round=n_rounds,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
        callbacks=callbacks,
    )
