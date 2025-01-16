#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


def _parse_args():
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument('--prediction', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    args = parser.parse_args()
    return args


def _load_target(args):
    target = pd.read_csv(args.target, index_col='Index').to_numpy()
    return target


def _load_prediction(args):
    prediction = pd.read_csv(args.prediction, index_col='Index')
    mu = prediction['Pred'].to_numpy()
    return mu


def _mape_score(mu, y):
    return abs(y - mu).mean() / y.max()


def _run_main():
    args = _parse_args()

    mu = _load_prediction(args)
    y = _load_target(args)

    scores = {
        'MAPE': _mape_score(mu, y).item(),
    }

    print(','.join([
        f'{name}={val:.3f}'
        for name, val in scores.items()
    ]))


if __name__ == '__main__':
    _run_main()
