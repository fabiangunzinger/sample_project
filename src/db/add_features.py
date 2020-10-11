import argparse
import os
import sqlite3
import sys

import pandas as pd

from .features import *
from src import config
from .registries import features_registry


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('sample')
    parser.add_argument('replace')
    return parser.parse_args()


def db_tables(connection):
    """List tables in database."""
    res = pd.read_sql("select name from sqlite_master", connection)
    return res.name.values


def add_features(sample):
    """Calculate features and add to database."""
    db_name = f'{sample}.db'
    db_path = os.path.join(config.DATADIR, db_name)
    conn = sqlite3.connect(db_path)
    data_name = f'data_{sample}.parquet'
    data_path = os.path.join(config.TEMPDIR, data_name)
    data = pd.read_parquet(data_path)
    db_tbls = db_tables(conn)
    for feature in features_registry:
        if feature.__name__ not in db_tbls:
            tbl = feature(data)
            tbl_name = 'ftr_' + feature.__name__
            tbl.to_sql(tbl_name, conn)


def main(argv=None):
    if argv is None:
        argv = sys.argv[:1]
    args = parse_args(argv)
    add_features(args.sample)


if __name__ == '__main__':
    sys.exit(main())
