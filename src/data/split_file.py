#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import contextlib
import os
import re
import sys


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('tempdir')
    return parser.parse_args()


def create_files(path, tempdir):
    """Create ten empty files with a header."""
    with open(path, 'rt') as f:
        header = f.readline()
        for n in range(1, 10):
            filepath = os.path.join(tempdir, f'{n}.csv')
            with open(filepath, 'wt') as file:
                file.write(header)


def add_lines(path, tempdir):
    """Add rows to files based on first digit of user id."""
    with open(path, 'rt') as source:
        with contextlib.ExitStack() as stack:
            files = {
                file.name[0]: stack.enter_context(open(file.path, 'at'))
                for file in os.scandir(tempdir)
                if file.name.endswith('.csv')
            }
            next(source)
            for line in source:
                user_id = re.findall('\d+', line)[1]
                files[user_id[0]].write(line)


def split_file(path, tempdir):
    create_files(path, tempdir)
    add_lines(path, tempdir)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    split_file(args.path, args.tempdir)


if __name__ == '__main__':
    sys.exit(main())
