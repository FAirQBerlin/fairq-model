from logging.config import dictConfig

from fairqmodel.db_connect import db_connect_source, get_query
from logging_config.logger_config import get_logger_config

dictConfig(get_logger_config())

with db_connect_source() as db:
    # db.execute("drop table if exists biweekly_features_all;")
    db.execute(get_query("biweekly_features_sampling_date_times"))
    db.execute(get_query("biweekly_features_no2"))
    db.execute(get_query("biweekly_features_no2_grid"))
    db.execute(get_query("biweekly_features_dwd"))
    db.execute(get_query("biweekly_features_stadtstruktur"))
    db.execute(get_query("biweekly_features_traffic"))
    db.execute(get_query("biweekly_features_all"))
