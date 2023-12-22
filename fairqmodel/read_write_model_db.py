import datetime
import logging
from logging.config import dictConfig
from typing import Optional

import pandas as pd

from fairqmodel.db_connect import db_connect_target, get_query, send_data_clickhouse
from fairqmodel.model_wrapper import ModelWrapper
from fairqmodel.read_write_model_aux_functions import model_as_str, model_from_str
from fairqmodel.time_handling import to_unix
from logging_config.logger_config import get_logger_config

dictConfig(get_logger_config())


def save_model_to_db(
    models: ModelWrapper,
    model_name: str,
    model_1_description: str,
    model_2_description: Optional[str],
    execution_time: datetime.datetime,
) -> int:
    """Stores a given model and corresponding information into the DB.

    :param models: ModelWrapper, Wrapper containing one or two models
    :param model_name: str, Specifies the type of model
    :param model_1_description: str, Description of the first model
    :param model_2_description: Optional[str], Description of second model if two stages are used
    :param execution_time: datetime.datetime, Date when the training was executed

    :return: int, model_id (under which the model was stored)
    """
    logging.info("Write model to DB")
    depvar = models.depvar

    # Get new id:
    with db_connect_target() as db:
        model_id = db.query_dataframe(get_query("model_id_max")).values.item() + 1

    assert models.model_1 is not None
    model_1_object_str = model_as_str(models.model_1)

    if models.use_two_stages:
        assert models.model_2 is not None
        model_2_object_str = model_as_str(models.model_2)
    else:
        model_2_object_str = None

    # Store as a DataFrame
    model_information = pd.DataFrame.from_dict(
        {
            "model_id": [model_id],
            "date_time_training_execution": [to_unix(execution_time)],
            "pollutant": [depvar],
            "model_name": [model_name],
            "description": [model_1_description],
            "model_object": [model_1_object_str],
            "description_residuals": [model_2_description],
            "model_object_residuals": [model_2_object_str],
        }
    )

    # Send model to DB
    send_data_clickhouse(df=model_information, table_name="model_description", mode="insert")

    return model_id


def retrieve_model_from_db(model_id: int) -> ModelWrapper:
    """Selects and loads a trained model(s) from the DB.

    :param model_id: int, id of the selected model

    :return: ModelWrapper, Wrapper object containing one or two trained xgb models
    """
    logging.info("Retrieve model from DB")
    # Get model_object from DB
    with db_connect_target() as db:
        dat_models = db.query_dataframe(get_query("model_by_id"), params={"id": model_id})

    if len(dat_models) > 1:
        try:
            with db_connect_target() as db:
                db.execute(get_query("optimize_model_description_final"))
        except:  # noqa
            raise

        raise ValueError("More than one model with the same id found in the database. Will be fixed in the next run.")

    model_object_str = dat_models["model_object"].item()
    model_object_residuals_str = dat_models["model_object_residuals"].item()

    model_1 = model_from_str(model_object_str)
    model_2 = model_from_str(model_object_residuals_str) if model_object_residuals_str is not None else None

    depvar = dat_models["pollutant"].item()

    # Build wrapper
    models = ModelWrapper(
        depvar=depvar,
        model_1=model_1,
        model_2=model_2,
    )

    return models


def write_model_to_models_final(depvar: str, domain: str, model_id: int) -> None:
    """
    Write the latest model ID to the database in case we want to use this new model as the productive model now.
    :param depvar: str, dependent variable (no2, pm10, or pm25)
    :param domain: str, temporal or spatial
    :param model_id: int, model ID
    """
    row_models_final = pd.DataFrame({"pollutant": [depvar], "domain": [domain], "model_id": [model_id]})
    send_data_clickhouse(df=row_models_final, table_name="models_final", mode="replace")
