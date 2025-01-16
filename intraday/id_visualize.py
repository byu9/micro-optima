#!/usr/bin/env python3
from argparse import ArgumentParser
from argparse import BooleanOptionalAction
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _parse_args():
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument('--prediction', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--parse-dates', action=BooleanOptionalAction)
    args = parser.parse_args()
    return args


def _load_target(args):
    target = pd.read_csv(args.target, index_col='Index')

    if args.parse_dates:
        target.index = pd.to_datetime(target.index, utc=True)
        target.sort_index(inplace=True)

    return target


def _load_prediction(args):
    prediction = pd.read_csv(args.prediction, index_col='Index')

    if args.parse_dates:
        prediction.index = pd.to_datetime(prediction.index, utc=True)
        prediction.sort_index(inplace=True)

    return prediction


def _run_main():
    args = _parse_args()

    prediction = _load_prediction(args)
    target = _load_target(args)

    plt.figure()
    plt.plot(target.index, target, label='target', marker='.')
    plt.plot(prediction.index, prediction, label='prediction', marker='+')
    plt.plot()
    plt.legend()
    plt.grid()
    plt.title(args.prediction)
    plt.xlabel('Observation Index')
    plt.xticks(rotation=45)
    plt.ylabel('Target')
    plt.show()


if __name__ == '__main__':
    _run_main()
