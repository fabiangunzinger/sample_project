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


def split_file(path, tempdir):
    """Split file into pieces based on first digit of user id."""
    DIGITS_RE = re.compile(r'\d+')
    filepaths = {n: os.path.join(tempdir, f'{n}.csv') for n in range(1, 10)}
    with open(path, 'rt') as source:
        with contextlib.ExitStack() as stack:
            files = {n: stack.enter_context(open(fp, 'a'))
                     for n, fp in filepaths.items()}
            header = source.readline()
            for f in files.values():
                f.write(header)
            for line in source:
                user_id = DIGITS_RE.findall(line)[1]
                first_digit = int(user_id[0])
                files[first_digit].write(line)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    split_file(args.path, args.tempdir)


if __name__ == '__main__':
    sys.exit(main())
