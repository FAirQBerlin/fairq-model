import logging
from logging.config import dictConfig
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from fairqmodel.model_parameters import get_xgboost_param
from logging_config.logger_config import get_logger_config

dictConfig(get_logger_config())


class ModelWrapper:
    def __init__(
        self,
        depvar: str,
        model_1: Optional[xgb.Booster] = None,
        model_2: Optional[xgb.Booster] = None,
        feature_cols_1: Optional[List[str]] = None,
        feature_cols_2: Optional[List[str]] = None,
        dev: bool = False,
    ) -> None:
        """Wrapper for xgb models to allow for a flexible use of either one or two models.
        An instance of the class can be created in one of two ways:
            1) Pre-trained model(s) is(are) are provided.
            2) Feature variables are provided on which new models can be trained.

        :param depvar: str, Dependent variable
        :param model_1: Optional[xgb.Booster], If provided, the first (pre-trained) model
        :param model_2: Optional[xgb.Booster], If provided, the second (pre-trained) model
        :param feature_cols_1: Optional[List[str]], Contains the features used for model_1
        :param feature_cols_2: Optional[List[str]], Contains the features used for model_2
        :param dev: bool, Specifies if development settings will be selected

        :return: None
        """

        # Assert general properties
        self._assert_properties(model_1, model_2, feature_cols_1, feature_cols_2)

        # Store provided parameters as attributes
        self.depvar = depvar
        self.model_1 = model_1
        self.model_2 = model_2
        self.feature_cols_1 = feature_cols_1
        self.feature_cols_2 = feature_cols_2
        self.dev = dev

        # Infer settings from provided parameters
        self.is_trained = self.model_1 is not None
        self.use_two_stages = (self.model_2 is not None) or (self.feature_cols_2 is not None)

        # Initialize available parameters
        # Either set feature_columns
        if self.is_trained:
            assert self.model_1 is not None  # for mypy checking
            self.feature_cols_1 = self.model_1.feature_names
            if self.use_two_stages:
                assert self.model_2 is not None
                self.feature_cols_2 = self.model_2.feature_names
        # Or initialize model parameters from .json
        else:
            self._retrieve_xgb_params()

        logging.info(
            "Model initialized for {} with: is_trained = {}, use_two_stages = {}".format(
                self.depvar, self.is_trained, self.use_two_stages
            )
        )

    def train(self, dat: pd.DataFrame) -> None:
        """Trains one or two models on the provided data.

        :param dat: pd.DataFrame, Containing the training data

        :return: None
        """
        if self.is_trained:
            logging.warning("Train function was called, but model is already trained")
            return
        logging.info("Start model training")

        # Train first stage
        dmatrix_1 = xgb.DMatrix(dat.loc[:, self.feature_cols_1], label=dat[self.depvar], enable_categorical=True)
        self.model_1 = self._train_model_1(dmatrix_1)

        # Train second stage
        if self.use_two_stages:
            self.model_2 = self._train_model_2(dat, dmatrix_1)

        # Update training status
        self.is_trained = True
        logging.info("Model was trained successfully")

    def predict(self, dat: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Performs the prediction with all available stages.
        If two stages are used, the second stage predicts the residual of the first stage,
        where residual = observation - first_prediction.

        :param dat: pd.DataFrame, Contains the data for predictions

        :return: pd.DataFrame, Predictions, estimated with the available models.
        """

        assert self.is_trained, "A model must be either loaded or trained before predictions can be performed."

        assert self.model_1 is not None
        dmatrix_1 = xgb.DMatrix(dat.loc[:, self.feature_cols_1], enable_categorical=True)
        prediction_stage_1 = self.model_1.predict(dmatrix_1)

        if not self.use_two_stages:
            prediction_total = prediction_stage_1
            prediction_stage_2 = None
        else:
            assert self.model_2 is not None
            dmatrix_2 = xgb.DMatrix(dat.loc[:, self.feature_cols_2], enable_categorical=True)
            prediction_stage_2 = self.model_2.predict(dmatrix_2)

            # Calculate total prediction from both stages and clip negative values to zero
            prediction_total = np.maximum((prediction_stage_1 + prediction_stage_2), 0)

        return prediction_total, prediction_stage_1, prediction_stage_2

    def _train_model_1(self, dmatrix_1: xgb.DMatrix) -> xgb.Booster:
        """Auxiliary function performing the model training of the first stage.

        :param dmatrix_1: xgb.DMatrix, Containing all variables of the first stage

        :return: xgb.Booster, The trained first stage
        """
        model_1 = xgb.train(
            params=self.xgb_param_1,
            dtrain=dmatrix_1,
            num_boost_round=self.n_rounds_1,
        )
        return model_1

    def _train_model_2(self, dat: pd.DataFrame, dmatrix_1: xgb.DMatrix) -> xgb.Booster:
        """Auxiliary function performing the model training of the second stage.
        The second stage predicts the residuals of the first stage where
        residual = observation - first_prediction.

        :param dat: pd.DataFrame, Containing the all data
        :param dmatrix_1: xgb.DMatrix, Containing all variables of the first stage

        :return: xgb.Booster, The trained second stage
        """
        assert self.model_1 is not None

        prediction = self.model_1.predict(dmatrix_1)
        residual = dat[self.depvar] - prediction
        dmatrix_2 = xgb.DMatrix(dat.loc[:, self.feature_cols_2], label=residual, enable_categorical=True)

        model_2 = xgb.train(
            params=self.xgb_param_2,
            dtrain=dmatrix_2,
            num_boost_round=self.n_rounds_2,
        )
        return model_2

    @staticmethod
    def _assert_properties(
        model_1: Optional[xgb.Booster] = None,
        model_2: Optional[xgb.Booster] = None,
        feature_cols_1: Optional[List[str]] = None,
        feature_cols_2: Optional[List[str]] = None,
    ) -> None:
        """Asserts that given parameters are valid combinations.

        :param model_1: Optional[xgb.Booster], If provided, the first (pre-trained) model
        :param model_2: Optional[xgb.Booster], If provided, the second (pre-trained) model
        :param feature_cols_1: Optional[List[str]], Contains the features used for model_1
        :param feature_cols_2: Optional[List[str]], Contains the features used for model_2

        :return: None
        """
        m1_not_none = model_1 is not None
        m2_not_none = model_2 is not None

        feat_1_not_none = feature_cols_1 is not None
        feat_2_not_none = feature_cols_2 is not None

        # Assert that only valid combinations of models and feature_cols are provided
        assert not (
            m1_not_none and feat_1_not_none
        ), "Invalid combination of parameters: Both model_1 and feature_cols_1 are provided."
        assert (
            m1_not_none or feat_1_not_none
        ), "Invalid combination of parameters: Neither model_1 nor feature_cols_1 is provided."
        assert not (
            m1_not_none and feat_2_not_none
        ), "Invalid combination of parameters: model_1 and feature_cols_2 are provided"
        assert not (
            m2_not_none and feat_1_not_none
        ), "Invalid combination of parameters: model_2 and feature_cols_1 are provided"

    def _retrieve_xgb_params(self) -> None:
        """Accesses model parameters from a .json file. Required for training new models.

        :return: None
        """
        self.xgb_param_1, self.n_rounds_1 = get_xgboost_param(self.depvar, self.use_two_stages, stage=1, dev=self.dev)

        if self.use_two_stages:
            self.xgb_param_2, self.n_rounds_2 = get_xgboost_param(
                self.depvar, self.use_two_stages, stage=2, dev=self.dev
            )
