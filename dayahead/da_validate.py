#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


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
    mu = prediction['Mean'].to_numpy()
    sigma = prediction['Std'].to_numpy()
    return mu, sigma


def _mape_score(mu, y):
    return abs(y - mu).mean() / y.max()


def _log_likelihood_score(mu, sigma, y):
    return -norm.logpdf(y, loc=mu, scale=sigma).mean()


def _crps_score(mu, sigma, y):
    z = (y - mu) / sigma
    score = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return score.mean()


def _run_main():
    args = _parse_args()

    mu, sigma = _load_prediction(args)
    y = _load_target(args)

    scores = {
        'MAPE': _mape_score(mu, y).item(),
        'LL': _log_likelihood_score(mu, sigma, y).item(),
        'CRPS': _crps_score(mu, sigma, y).item()
    }

    print(','.join([
        f'{name}={val:.3f}'
        for name, val in scores.items()
    ]))


if __name__ == '__main__':
    _run_main()
