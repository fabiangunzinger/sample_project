import os
import sqlite3
import pandas as pd
from src import config


def feature_table(sample, experiment, order=False):
    """Create dataframe with target and features.

    Merge two tables as workaround of SQLite's 2,500 columns limit.
    """
    db_path = config.DATADIR
    db_name = f'features_{sample}.db'
    db = os.path.join(db_path, db_name)
    conn = sqlite3.connect(db)

    target = experiment
    features = [
        'entropy',
        'grocery_shop_freq',
        'pct_credit',
        'pct_manual_tags',
        'merchant_spending_shares',
        'tag_spending_shares',
        'merchant_dummies',
        'tag_dummies',
    ]

    select = f'select * from {target} '
    joins = ' '.join([f'join {f} using(user_id)' for f in features[:-2]])
    table1 = pd.read_sql_query(select + joins, conn).set_index('user_id')

    select = f'select * from {features[-2]} '
    joins = ' '.join([f'join {f} using(user_id)' for f in features[-1:]])
    table2 = pd.read_sql_query(select + joins, conn).set_index('user_id')

    result = table1.join(table2)

    # move dense features to front
    if order:
        sparse = ['regyear', 'merch', 'merchshare', 'tag', 'tagshare']
        regex = '|'.join(sparse)
        last = result.columns[result.columns.str.contains(regex)]
        first = set(result.columns) - set(last)
        result = result[list(first) + list(last)]

    return result
