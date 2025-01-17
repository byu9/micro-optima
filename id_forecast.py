#!/usr/bin/env python3
import pickle
from abc import ABCMeta
from abc import abstractmethod
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

_forecast_horizon = 4


class IDForecastModel(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, target, save_prediction=None):
        raise NotImplementedError

    @abstractmethod
    def predict(self, target, save_prediction=None):
        raise NotImplementedError

    @staticmethod
    def _build_prior_target_matrix(target, look_back):
        feature = pd.DataFrame(index=target.index, data={
            f'Prior{shift}': target['Target'].shift(shift)
            for shift in range(1, look_back + 1)
        }).dropna(axis='index')
        return feature

    @staticmethod
    def _build_future_target_matrix(target, look_ahead):
        target = pd.DataFrame(index=target.index, data={
            f'Target{shift}': target['Target'].shift(-shift)
            for shift in range(look_ahead)
        }).dropna(axis='index')
        return target

    def save_model(self, filename):
        with open(filename, 'wb') as model_file:
            pickle.dump(self, file=model_file)

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as model_file:
            model = pickle.load(model_file)

        if not isinstance(model, IDForecastModel):
            raise RuntimeError(f'"{model}" is not a valid intra-day forecast model.')

        return model

    @staticmethod
    def _save_prediction(data, index, filename):
        prediction = pd.DataFrame(index=index, data=data)
        header = [f'Pred{shift}' for shift in range(_forecast_horizon)]
        prediction.to_csv(filename, index_label='Index', header=header)


class ScikitModelBase(IDForecastModel, metaclass=ABCMeta):
    _LOOK_BACK = None

    def __init__(self):
        self._model = None

    def fit(self, target, save_prediction=None):
        prior_target = self._build_prior_target_matrix(target, look_back=self._LOOK_BACK)
        future_target = self._build_future_target_matrix(target, look_ahead=_forecast_horizon)

        indices = prior_target.index.intersection(future_target.index)
        prior_target = prior_target.reindex(indices)
        future_target = future_target.reindex(indices)

        self._model.fit(prior_target, future_target)
        predict_data = self._model.predict(prior_target)

        if save_prediction is not None:
            self._save_prediction(predict_data, index=prior_target.index, filename=save_prediction)

        return predict_data

    def predict(self, target, save_prediction=None):
        prior_target = self._build_prior_target_matrix(target, look_back=self._LOOK_BACK)
        predict_data = self._model.predict(prior_target)

        if save_prediction is not None:
            self._save_prediction(predict_data, index=prior_target.index, filename=save_prediction)

        return predict_data


class DT8(ScikitModelBase):
    _LOOK_BACK = 8

    def __init__(self):
        super().__init__()
        self._model = DecisionTreeRegressor()


class RF8(ScikitModelBase):
    _LOOK_BACK = 8

    def __init__(self):
        super().__init__()
        self._model = RandomForestRegressor()


class AR8(ScikitModelBase):
    _LOOK_BACK = 8

    def __init__(self):
        super().__init__()
        self._model = LinearRegression()


class NN8K20(ScikitModelBase):
    _LOOK_BACK = 8

    def __init__(self):
        super().__init__()
        self._model = KNeighborsRegressor(n_neighbors=20)


class MLP111(ScikitModelBase):
    _LOOK_BACK = 8

    def __init__(self):
        super().__init__()
        self._model = MLPRegressor(
            hidden_layer_sizes=(100,100,100),
            learning_rate='adaptive',
            max_iter=1000
        )


_supported_models = {
    'ar8': AR8,
    'dt8': DT8,
    'nn8k20': NN8K20,
    'rf8': RF8,
    'mlp111': MLP111
}


def _parse_args():
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument('--learn', type=str, default=None)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--prediction', type=Path, default=None)
    args = parser.parse_args()

    if args.learn is not None:
        if args.learn not in _supported_models:
            raise ValueError(f'Model "{args.learn}" is not supported.')

        if args.prediction is None:
            raise ValueError(f'Prediction file must be specified in forecast mode.')

    return args


def _load_target(filename):
    target = pd.read_csv(filename, index_col='Index')
    return target


def _run_learn_task(args):
    target = _load_target(args.target)
    model = _supported_models[args.learn]()
    model.fit(target, save_prediction=args.prediction)
    model.save_model(args.model)


def _run_predict_task(args):
    target = _load_target(args.target)
    model = IDForecastModel.load_model(args.model)
    model.predict(target, save_prediction=args.prediction)


def _run_main():
    args = _parse_args()

    if args.learn:
        _run_learn_task(args)

    else:
        _run_predict_task(args)


if __name__ == '__main__':
    _run_main()
