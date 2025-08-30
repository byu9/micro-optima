#!/usr/bin/env python3
from argparse import ArgumentParser
from argparse import BooleanOptionalAction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


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
        target.sort_index(inplace=True)

    return target


def _load_prediction(args):
    prediction = pd.read_csv(args.prediction, index_col='Index')

    if args.parse_dates:
        prediction.index = pd.to_datetime(prediction.index, utc=True)
        prediction.sort_index(inplace=True)

    predict_data = norm(loc=prediction['Mean'], scale=prediction['Std'])

    return predict_data


def _plot_heatmap(dist, vlo, vhi, hlo, hhi, resolution=200):
    vertical = np.linspace(vlo, vhi, resolution).reshape(-1, 1)
    density = dist.pdf(vertical)
    plt.imshow(density, extent=(hlo, hhi, vlo, vhi), aspect='auto', origin='lower', cmap='hot')
    plt.colorbar(label='Probability Density')


def _run_main():
    args = _parse_args()

    prediction = _load_prediction(args)
    target = _load_target(args)

    plt.figure(figsize=(12, 6))
    plt.scatter(target.index, target, label='target', marker='.', s=1, color='white')
    _plot_heatmap(prediction,
                  vlo=target['Target'].min(),
                  vhi=target['Target'].max(),
                  hlo=target.index.min(),
                  hhi=target.index.max())
    plt.legend()
    plt.grid(color='white', alpha=0.3)
    plt.title(args.prediction)
    plt.xlabel('Observation Index')
    plt.xticks(rotation=45)
    plt.ylabel('Target')
    plt.tight_layout()

    if args.save is None:
        plt.show()

    else:
        plt.savefig(args.save)


if __name__ == '__main__':
    _run_main()
