import os
import pandas as pd


def export_latex_table(table, name, path=None, **kwargs):
    if path is None:
        icloud = '/Users/fgu/Library/Mobile Documents/com~apple~CloudDocs/'
        proj = 'fab/projects/habits/'
        subdir = 'output/tables'
        path = os.path.join(icloud, proj, subdir)
    with pd.option_context('max_colwidth', None):
        with open(os.path.join(path, name), 'w') as f:
            f.write(table.to_latex(index=False, escape=False, **kwargs))


def info(df):
    """Print basic dataframe info."""
    pad = 15
    usrs = df.user_id.nunique()
    rows, cols = df.shape
    print(f"Users: {usrs:>{pad-len('users')},}")
    print(f"Rows: {rows:>{pad-len('rows')},}")
    print(f"Cols: {cols:>{pad-len('cols')},}")


def reorder(df, columns):
    """Place supplied columns to front of dataframe."""
    old_order = list(df.columns)
    for col in columns:
        old_order.remove(col)
    new_order = columns + old_order
    return df[new_order]


def user_data(df, user_id):
    """Return data for user."""
    return df[df.user_id.eq(user_id)]


def txn_data(df, txns):
    """Return dataframe with supplied transactions."""
    return df[df.transaction_id.isin(txns)].copy()


def tag_data(df, tag, var='auto_tag'):
    """Return data with specified tag."""
    return df[df[var].eq(tag)]
