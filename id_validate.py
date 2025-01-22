#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd


def _mape_score(prediction, target):
    return abs(prediction - target).mean(axis='index') / abs(target).max() * 100


def _mae_score(prediction, target):
    return abs(prediction - target).mean(axis='index')


def _mse_score(prediction, target):
    return np.square(prediction - target).mean(axis='index')


def _build_future_target_matrix(target, look_ahead):
    target = pd.DataFrame(index=target.index, data={
        f'Pred{shift}': target['Target'].shift(-shift)
        for shift in range(look_ahead)
    }).dropna(axis='index')

    return target


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
    target = pd.read_csv(filename, index_col='Index')
    return target


def _load_prediction(filename):
    prediction = pd.read_csv(filename, index_col='Index')
    return prediction


def _calculate_score(predict_filename, target_filename, score_func):
    prediction = _load_prediction(predict_filename)
    target_data = _load_target(target_filename)
    target = _build_future_target_matrix(target_data, look_ahead=len(prediction.columns))
    target = target.reindex(prediction.index)

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
