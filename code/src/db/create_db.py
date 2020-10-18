#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sqlite3
import sys
import pandas as pd
from src import config


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('sample')
    parser.add_argument('replace')
    return parser.parse_args()


def db_tables(connection):
    """List tables in database."""
    res = pd.read_sql("select name from sqlite_master", connection)
    return res.name.values


def create_database(sample):
    """Create database with tables for targets, outcomes, and predictions."""
    db_name = f'{sample}.db'
    db_path = os.path.join(config.DATADIR, db_name)
    conn = sqlite3.connect(db_path)
    usr_name = f'users_{sample}.csv'
    usr_path = os.path.join(config.DATADIR, usr_name)
    users = pd.read_csv(usr_path)
    db_tbls = db_tables(conn)
    for tbl in ['decisions', 'outcomes', 'predictions']:
        if tbl not in db_tbls:
            users.to_sql(tbl, conn, index=False)
            conn.execute(f"create index idx_{tbl}_user_id on {tbl}(user_id)")


def main(argv=None):
    if argv is None:
        argv = sys.argv[:1]
    args = parse_args(argv)
    create_database(args.sample)


if __name__ == '__main__':
    sys.exit(main())
