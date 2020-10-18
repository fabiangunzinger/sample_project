import argparse
import os
import sqlite3
import sys
import pandas as pd
from .decisions import *
from src import config
from .registries import decisions_registry


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('sample')
    parser.add_argument('replace')
    return parser.parse_args()


def table_cols(table, conn):
    """List table columns."""
    res = pd.read_sql(f"select name from pragma_table_info('{table}');", conn)
    return res.name.values


def add_column(column, column_name, table, conn):
    """Add column to table."""
    column.to_sql('tmp', conn, index=False)
    conn.executescript(
        f"""
        alter table {table} add column {column_name};

        update {table}
        set {column_name} = (
            select {column_name} from tmp
            where {table}.user_id = tmp.user_id);

        drop table tmp;
        """)


def add_decisions(sample):
    """Calculate decision variables and add to database."""
    db_name = f'{sample}.db'
    db_path = os.path.join(config.DATADIR, db_name)
    conn = sqlite3.connect(db_path)
    data_name = f'data_{sample}.parquet'
    data_path = os.path.join(config.TEMPDIR, data_name)
    data = pd.read_parquet(data_path)
    tbl_cols = table_cols('decisions', conn)
    for decision in decisions_registry:
        if decision.__name__ not in tbl_cols:
            col = decision(data)
            col_name = decision.__name__
            add_column(col, col_name, 'decisions', conn)


def main(argv=None):
    if argv is None:
        argv = sys.argv[:1]
    args = parse_args(argv)
    add_decisions(args.sample)


if __name__ == '__main__':
    sys.exit(main())
