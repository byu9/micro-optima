#!/usr/bin/env python3
import pickle
from argparse import ArgumentParser
from argparse import BooleanOptionalAction
from pathlib import Path

import pandas as pd

from fuzzyprob import FuzzyProbTree


def _parse_args():
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument('--learn', default=False, action=BooleanOptionalAction)
    parser.add_argument('--feature', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--target', type=Path, default=None)
    parser.add_argument('--prediction', type=Path, default=None)
    args = parser.parse_args()

    if args.target is None:
        if args.learn:
            raise ValueError('Target file must be specified in learning mode.')

    if args.prediction is None:
        if not args.learn:
            raise ValueError('Prediction file must be specified in prediction mode.')

    return args


def _write_prediction_file(model: FuzzyProbTree, feature, filename):
    predict_dist = model.predict(feature)
    prediction = pd.DataFrame(index=feature.index, data={
        'Mean': predict_dist.mean(),
        'Std': predict_dist.std()
    })
    prediction.to_csv(filename)


def _load_feature(filename):
    feature = pd.read_csv(filename, index_col='Index').sort_index()
    return feature


def _load_target(filename):
    target = pd.read_csv(filename, index_col='Index').sort_index()
    return target


def _run_learn_task(args):
    feature = _load_feature(args.feature)
    target = _load_target(args.target)

    if not feature.index.equals(target.index):
        raise ValueError(f'Feature and target contains different indices.')

    model = FuzzyProbTree(max_split=20, batch_size=16, epochs=0, min_samples=5)
    model.fit(feature.to_numpy(), target.to_numpy().squeeze())

    with open(args.model, 'wb') as model_file:
        pickle.dump(model, file=model_file)

    if args.prediction is not None:
        _write_prediction_file(model, feature, filename=args.prediction)


def _run_predict_task(args):
    with open(args.model, 'rb') as model_file:
        model = pickle.load(model_file)

    feature = _load_feature(args.feature)
    _write_prediction_file(model, feature, filename=args.prediction)


def _run_main():
    args = _parse_args()

    if args.learn:
        _run_learn_task(args)

    else:
        _run_predict_task(args)


if __name__ == '__main__':
    _run_main()
