import argparse
import os
import sys
import pandas as pd
from src import config
from src.helpers.helpers import export_latex_table


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('sample')
    return parser.parse_args()


def sumstats_table(df, varlist, pctls=None, cols=None):
    """Produce summary statistics at user level."""
    if pctls is None:
        pctls = [.1, .25, .5, .75, .9]
    if cols is None:
        cols = ['mean', '10%', '25%', '50%', '75%', '90%']

    g = df.groupby('user_id')
    tbl = pd.DataFrame()

    for var in varlist:
        stats = g[var].nunique().describe(percentiles=pctls)
        frame = stats.to_frame().T
        tbl = tbl.append(frame)
    tbl.index = ['Banks', 'Accounts']
    tbl = tbl[cols].reset_index()
    tbl.columns = ['', 'Mean', 'p10', 'p25', 'p50', 'p75', 'p90']
    return tbl


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    file = f'data_{args.sample}.parquet'
    path = os.path.join(config.TEMPDIR, file)
    df = pd.read_parquet(path)
    varlist = ['bank', 'account_id']
    tbl = sumstats_table(df, varlist)
    export_latex_table(tbl, name='sumstats.tex')


if __name__ == '__main__':
    sys.exit(main())
