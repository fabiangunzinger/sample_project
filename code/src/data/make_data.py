#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import tempfile
import pandas as pd
from src import config
from src.helpers.helpers import export_latex_table
from src.data import (
    count,
    split_file,
    read_raw,
    select_sample,
    clean_data,
    selection_table
)


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('sample')
    return parser.parse_args()


def make_data(sample):
    """Produce clean dataset."""
    with tempfile.TemporaryDirectory() as tempdir:
        fp = os.path.join(config.TEMPDIR, f'data_{sample}.csv')
        print('Splitting data...')
        split_file(fp, tempdir)
        raw_pieces = (f.path for f in os.scandir(tempdir)
                      if f.name.endswith('.csv'))
        clean_pieces = []
        for piece in raw_pieces:
            print(os.path.basename(piece))
            clean_piece = (
                read_raw(piece)
                .pipe(clean_data)
                .pipe(select_sample)
            )
            clean_pieces.append(clean_piece)
        clean_name = f'data_{sample}.parquet'
        clean_path = os.path.join(config.TEMPDIR, clean_name)
        clean = pd.concat(clean_pieces)
        clean.to_parquet(clean_path)
        users_name = f'users_{sample}.csv'
        users_path = os.path.join(config.DATADIR, users_name)
        users = pd.Series(sorted(clean.user_id.unique()))
        users.to_csv(users_path, index=False)


def main(argv=None):
    if argv is None:
        argv = sys.argv[:1]
    args = parse_args(argv)
    make_data(args.sample)
    tbl = selection_table(count)
    tbl_name = f'sample_selection_{args.sample}.tex'
    export_latex_table(tbl, name=tbl_name, column_format='lrrrr')
    print(tbl)


if __name__ == '__main__':
    sys.exit(main())
