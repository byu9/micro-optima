#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def _mape_score(mu, y):
    return abs(y - mu).mean() / y.max()


def _log_likelihood_score(mu, sigma, y):
    return -norm.logpdf(y, loc=mu, scale=sigma).mean()


def _crps_score(mu, sigma, y):
    z = (y - mu) / sigma
    score = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return score.mean()


def _parse_args():
    parser = ArgumentParser(
        allow_abbrev=False,
        description='Validates intraday predictions against actual observations.'
    )
    parser.add_argument(
        '--pairs', type=Path, required=True,
        help='CSV file listing prediction/target file pairs'
    )
    parser.add_argument(
        '--scoreboard', type=Path, required=True,
        help='file to write the validation scores to'
    )
    args = parser.parse_args()
    return args


def _load_pairs_table(filename):
    pairs = pd.read_csv(filename, index_col='Label')
    return pairs.to_dict(orient='index')


def _load_target(filename):
    target = pd.read_csv(filename, index_col='Index').to_numpy()
    return target


def _load_prediction(filename):
    prediction = pd.read_csv(filename, index_col='Index')
    mu = prediction['Mean'].to_numpy()
    sigma = prediction['Std'].to_numpy()
    return mu, sigma


def _calculate_score(predict_filename, target_filename, score_func):
    prediction = _load_prediction(predict_filename)
    target = _load_target(target_filename)
    score = score_func(prediction=prediction, target=target)
    return score


def _run_main():
    args = _parse_args()
    pairs = _load_pairs_table(args.pairs)

    score_dict = {
        label: _calculate_score(paths['Prediction'], paths['Target'], _mape_score)
        for label, paths in pairs.items()
    }

    scoreboard = pd.DataFrame.from_dict(score_dict, orient='index')
    scoreboard.loc['(mean)'] = scoreboard.mean(axis='index')
    scoreboard.to_csv(args.scoreboard, index_label='Label')


if __name__ == '__main__':
    _run_main()
