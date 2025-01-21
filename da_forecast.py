#!/usr/bin/env python3
import pickle
from abc import ABCMeta
from abc import abstractmethod
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from ngboost import NGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

from fuzzyprob import FuzzyProbTree as FuzzyProbTree


class DAForecastModel(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, feature, target, save_prediction=None):
        raise NotImplementedError

    @abstractmethod
    def predict(self, feature, save_prediction=None):
        raise NotImplementedError

    def save_model(self, filename):
        with open(filename, 'wb') as model_file:
            pickle.dump(self, file=model_file)

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as model_file:
            model = pickle.load(model_file)

        if not isinstance(model, DAForecastModel):
            raise RuntimeError(f'"{model}" is not a valid intra-day forecast model.')

        return model

    @staticmethod
    def _save_prediction(mean, std, index, filename):
        prediction = pd.DataFrame(index=index, data={
            'Mean': mean,
            'Std': std
        })
        prediction.to_csv(filename, index_label='Index')


class NGBoost(DAForecastModel):
    def __init__(self):
        super().__init__()
        self._model = None

    def fit(self, feature, target, save_prediction=None):
        self._model = NGBRegressor()
        self._model.fit(feature.to_numpy(), target.to_numpy().squeeze())
        return self.predict(feature, save_prediction=save_prediction)

    def predict(self, feature, save_prediction=None):
        predict_data = self._model.predict_dist(feature.to_numpy())

        if save_prediction is not None:
            self._save_prediction(mean=predict_data.mean(), std=predict_data.std(),
                                  index=feature.index, filename=save_prediction)
        return predict_data


class FuzzyProb(DAForecastModel):

    def __init__(self):
        super().__init__()
        self._model = None

    def fit(self, feature, target, save_prediction=None):
        self._model = FuzzyProbTree(max_split=20, batch_size=16, epochs=0, min_samples=5)
        self._model.fit(feature.to_numpy(), target.to_numpy().squeeze())
        return self.predict(feature, save_prediction=save_prediction)

    def predict(self, feature, save_prediction=None):
        predict_data = self._model.predict(feature.to_numpy())

        if save_prediction is not None:
            self._save_prediction(mean=predict_data.mean(), std=predict_data.std(),
                                  index=feature.index, filename=save_prediction)
        return predict_data


class GPR(DAForecastModel):
    def __init__(self):
        super().__init__()
        self._model = None

    def fit(self, feature, target, save_prediction=None):
        self._model = GaussianProcessRegressor()
        self._model.fit(feature, target)
        return self.predict(feature, save_prediction=save_prediction)

    def predict(self, feature, save_prediction=None):
        predict_mean, predict_std = self._model.predict(feature, return_std=True)

        if save_prediction is not None:
            self._save_prediction(mean=predict_mean, std=predict_std,
                                  index=feature.index, filename=save_prediction)
        return predict_mean, predict_std


_supported_models = {
    'fuzzyprob': FuzzyProb,
    'gpr': GPR,
    'ngboost': NGBoost,
}


def _parse_args():
    parser = ArgumentParser(
        allow_abbrev=False,
        description='Learns a day-ahead forecast model or uses a learned model to make a forecast.'
    )

    parser.add_argument(
        '--learn', type=str, default=None,
        choices=_supported_models.keys(),
        help='put the program in learn mode and learn with specified model'
    )

    parser.add_argument(
        '--model', type=Path, required=True,
        help='in learn mode, the file path to write the model to; '
             'in forecast mode, the file path to load the model from'
    )

    parser.add_argument(
        '--feature', type=Path, required=True,
        help='the file path to load historical covariates from'
    )

    parser.add_argument(
        '--target', type=Path, required=True,
        help='the file path to load historical observations from'
    )

    parser.add_argument(
        '--prediction', type=Path, default=None,
        help='the file path to write predictions to'
    )

    args = parser.parse_args()

    if args.learn is not None:
        if args.learn not in _supported_models:
            raise ValueError(f'Model "{args.learn}" is not supported.')

        if args.prediction is None:
            raise ValueError(f'Prediction file must be specified in forecast mode.')

    return args


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

    model = _supported_models[args.learn]()
    model.fit(feature, target, save_prediction=args.prediction)
    model.save_model(args.model)


def _run_predict_task(args):
    feature = _load_feature(args.feature)
    model = DAForecastModel.load_model(args.model)
    model.predict(feature, save_prediction=args.prediction)


def _run_main():
    args = _parse_args()

    if args.learn:
        _run_learn_task(args)

    else:
        _run_predict_task(args)


if __name__ == '__main__':
    _run_main()
