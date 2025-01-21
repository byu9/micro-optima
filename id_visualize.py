#!/usr/bin/env python3
from argparse import ArgumentParser
from argparse import BooleanOptionalAction
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _parse_args():
    parser = ArgumentParser(
        allow_abbrev=False,
        description='Visualizes the prediction of a intraday forecast model.'
    )
    parser.add_argument(
        '--prediction', type=Path, required=True,
        help='path to load the prediction file from'
    )
    parser.add_argument(
        '--target', type=Path, required=True,
        help='path to load the target file from'
    )
    parser.add_argument(
        '--parse-dates', action=BooleanOptionalAction,
        help='parse the index column as timestamps instead of observation numbers'
    )
    parser.add_argument(
        '--title', type=str, default=None,
        help='use the given title in the plot'
    )
    parser.add_argument(
        '--save', type=Path, default=None,
        help='save the plot to the given path as image (image type determined from suffix)'
    )
    args = parser.parse_args()
    return args


def _load_target(args):
    target = pd.read_csv(args.target, index_col='Index')
    if args.parse_dates:
        target.index = pd.to_datetime(target.index, utc=True)

    return target


def _load_prediction(args):
    prediction = pd.read_csv(args.prediction, index_col='Index')

    if args.parse_dates:
        prediction.index = pd.to_datetime(prediction.index, utc=True)

    return prediction


def _build_future_target_matrix(target, look_ahead):
    target = pd.DataFrame(index=target.index, data={
        f'Pred{shift}': target['Target'].shift(-shift)
        for shift in range(look_ahead)
    }).dropna(axis='index')

    return target


def _run_main():
    args = _parse_args()
    target = _load_target(args)
    prediction = _load_prediction(args)
    target = _build_future_target_matrix(target, look_ahead=len(prediction.columns))

    ax = target.plot(
        subplots=True,
        linestyle='none',
        marker='+',
        ms=1,
        legend=False,
        color='black'
    )

    prediction.plot(
        ax=ax,
        subplots=True,
        grid=True,
        legend=False,
        color='red',
        alpha=0.5,
        title=args.title
    )

    if args.save is None:
        plt.show()

    else:
        plt.savefig(args.save)


if __name__ == '__main__':
    _run_main()
