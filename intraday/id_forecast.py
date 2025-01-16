#!/usr/bin/env python3
import pickle
from argparse import ArgumentParser
from argparse import BooleanOptionalAction
from pathlib import Path

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA as _ARIMA

_forecast_horizon = 4


def _parse_args():
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument('--learn', default=False, action=BooleanOptionalAction)
    parser.add_argument('--in-sample', default=False, action=BooleanOptionalAction)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--prediction', type=Path, default=None)
    args = parser.parse_args()

    if args.learn is not None:
        if args.prediction is None:
            raise ValueError(f'Prediction file must be specified in forecast mode.')

    return args


class ARIMA800:
    def __init__(self):
        self._model = None

    def fit(self, target):
        self._model = _ARIMA(target.reset_index(drop=True), order=(8, 0, 0)).fit()
        predict = self._model.predict()
        predict.index = target.index
        return predict

    def predict(self, target, in_sample: bool):
        model = self._model.apply(target.reset_index(drop=True))

        if in_sample:
            predict = model.predict()
            predict.index = target.index

        else:
            predict = model.forecast(_forecast_horizon)

        return predict


def _save_model(model, filename):
    with open(filename, 'wb') as model_file:
        pickle.dump(model, file=model_file)


def _load_model(filename):
    with open(filename, 'rb') as model_file:
        model = pickle.load(model_file)

    return model


def _load_target(filename):
    target = pd.read_csv(filename, index_col='Index')
    return target


def _save_prediction(prediction, filename):
    prediction.to_csv(filename, index_label='Index', header=['Pred'])


def _run_learn_task(args):
    target = _load_target(args.target)

    model = ARIMA800()
    prediction = model.fit(target)

    _save_model(model, filename=args.model)

    if args.prediction is not None:
        _save_prediction(prediction, filename=args.prediction)


def _run_predict_task(args):
    target = _load_target(args.target)
    model = _load_model(args.model)

    prediction = model.predict(target, in_sample=args.in_sample)
    _save_prediction(prediction, filename=args.prediction)


def _run_main():
    args = _parse_args()

    if args.learn:
        _run_learn_task(args)

    else:
        _run_predict_task(args)


if __name__ == '__main__':
    _run_main()
