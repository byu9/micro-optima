#!/usr/bin/env python3
from argparse import ArgumentParser
from argparse import BooleanOptionalAction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


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


def _plot_heatmap(dist, vlo, vhi, hlo, hhi, resolution=200):
    vertical = np.linspace(vlo, vhi, resolution).reshape(-1, 1)
    density = dist.pdf(vertical)
    plt.imshow(density, extent=(hlo, hhi, vlo, vhi), aspect='auto', origin='lower', cmap='hot')
    plt.colorbar(label='Probability Density')


def _run_main():
    args = _parse_args()

    prediction = _load_prediction(args)
    predict_dist = norm(loc=prediction['Mean'], scale=prediction['Std'])
    target = _load_target(args)

    plt.figure()
    plt.scatter(target.index, target, label='target', marker='.', s=10, color='white')
    _plot_heatmap(predict_dist,
                  vlo=predict_dist.ppf(0.05).min(),
                  vhi=predict_dist.ppf(0.95).max(),
                  hlo=prediction.index.min(),
                  hhi=prediction.index.max())
    plt.legend()
    plt.grid(color='white', alpha=0.3)
    plt.title(args.prediction)
    plt.xlabel('Observation Index')
    plt.ylabel('Target')
    plt.show()


if __name__ == '__main__':
    _run_main()
