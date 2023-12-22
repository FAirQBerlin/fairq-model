import logging
import multiprocessing
import os
from logging.config import dictConfig
from pathlib import Path
from typing import Optional

import pandas as pd
from clickhouse_driver import Client
from dotenv import load_dotenv
from retrying import retry

from logging_config.logger_config import get_logger_config

load_dotenv()
dictConfig(get_logger_config())


def mode() -> str:
    env = os.getenv("MODE", "DEV")  # default mode is "DEV"
    return env


def db_suffix() -> str:
    env = mode()
    db_suffix = {"DEV": "", "PROD": "prod_"}
    return db_suffix[env]


def db_connect_source() -> Client:
    """
    Return Client object for db connection to clickhouse.
    :return: Client object for db connection to clickhouse
    """
    max_threads = max(multiprocessing.cpu_count() - 4, 1)
    return Client(
        host=os.getenv("DB_HOST"),
        port=None,
        database=os.getenv("DB_SOURCE"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        secure=True,
        settings={"use_numpy": True, "max_threads": max_threads},
    )


def db_connect_target() -> Client:
    """
    Return Client object for db connection to clickhouse.
    :return: Client object for db connection to clickhouse
    """
    max_threads = max(multiprocessing.cpu_count() - 4, 1)
    return Client(
        host=os.getenv("DB_HOST"),
        port=None,
        database=os.getenv("DB_TARGET"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        secure=True,
        settings={"use_numpy": True, "max_threads": max_threads},
    )


def get_query(query: str, parametrized_tables: Optional[dict] = None) -> str:
    """
    Return query result as string.

    :param query: query string
    :param parametrized_tables: optional dictionary of tables which are parametrized in the query via {table_name}

    :return: query result as pandas DataFrame
    """
    if ";" in query:
        return query

    # load query from file with the name "query" from the /sql directory
    file_name = query if ".sql" in query else f"{query}.sql"

    query_file_path = Path(__file__).parent / "sql" / file_name

    with open(query_file_path, "r") as f:
        query = f.read()

    query = query.replace("fairq_output.", f"fairq_{db_suffix()}output.")
    query = query.replace("fairq_features.", f"fairq_{db_suffix()}features.")

    if parametrized_tables is not None:
        query = query.format(**parametrized_tables)

    if "{" in query:
        ValueError(f"There is a parameter left in the query that could not be substituted by {parametrized_tables}")

    return query


@retry(stop_max_attempt_number=3, wait_fixed=60000)
def send_data_clickhouse(
    df: pd.DataFrame,
    table_name: str,
    mode: str = "replace",
    schema_name: str = os.getenv("DB_TARGET") or "fairq_output",
) -> bool:
    """
    Send data of a given df to clickhouse.
    :param df: DataFrame, Containing data to write to db
    :param mode: "insert", "replace". "insert" just inserts the data.
    "replace" inserts the data and then optimizes the table to remove
    all duplicates w.r.t. the order statement. Only allowed if the table engine is ReplacingMergeTree.
    Default in the repo is "replace".
    :param schema_name: name of db schema, fairq_output per default if env "DB_TARGET" not set
    :param table_name: name of db table
    """
    if mode not in ["insert", "replace"]:
        raise ValueError("Allowed modes are: insert, replace")

    if mode == "replace":
        check_for_replacing_merge_tree(table_name, schema_name)
    logging.info(f"Write DataFrame with shape of {df.shape} to DB")
    if df.shape[0] > 0:
        with db_connect_target() as db:
            try:
                logging.info("Sending data to database {}@{}".format(table_name, schema_name))
                db.insert_dataframe(f"INSERT INTO {schema_name}.{table_name} VALUES", df)
            except:  # noqa
                raise

            finally:
                optimizing_table_and_mv(db, table_name, schema_name, mode)

    logging.info("Data sent successfully \n")

    return True


def optimizing_table_and_mv(db: Client, table_name: str, schema_name: str, mode: str):
    """ "
    Optimize table and materialized view (mv) after insert.
    :db: Client object for db connection to clickhouse
    :param table_name: name of the table
    :param schema_name: name of the database schema
    :param mode: "replace". data is only optimized if mode is "replace",
    mv is only optimized if exists
    """
    if mode == "replace":
        logging.info("Optimizing table to remove duplicates...")
        db.execute(f"Optimize table {schema_name}.{table_name} final;")

    if materialized_view_exists(table_name, schema_name):
        logging.info("Optimize table processed by materialized view...")
        db.execute(f"Optimize table {schema_name}.{table_name}_processed final;")


def check_for_replacing_merge_tree(table_name: str, schema_name: str):
    """
    Check if target table has engine "check_for_replacing_merge_tree"; raise error if not.
    :param table_name: name of the table
    :param schema_name: name of the database schema
    """
    with db_connect_target() as db:
        logging.info("Checking if table engine is 'ReplacingMergeTree'...")
        table_engine = db.execute(
            f"SELECT engine FROM system.tables where database = '{schema_name}' and name = '{table_name}';"
        )[0][0]
    if table_engine != "ReplacingMergeTree":
        raise Exception(
            f"Can't use mode 'replace' for table {table_name} since as table engine is not ReplacingMergeTree."
        )


def materialized_view_exists(table_name: str, schema_name: str):
    """
    Check if target table for materialized view exists, so
    it can be optimized after insert as well.
    :param table_name: name of the table
    :param schema_name: name of the database schema
    """
    with db_connect_target() as db:
        logging.info("Checking if materialized view exists ...")
        mv_exists = db.execute(f"exists {schema_name}.{table_name}_processed;")[0][0]
    return mv_exists == 1


# __file__ = 'fairqmodel/db_connect.py'
