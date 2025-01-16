#!/usr/bin/env python3
import logging
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _download_file(url, filename):
    if not Path(filename).is_file():
        _logger.info(f'Retrieving {url} as "{filename}".')
        urlretrieve(url, filename=filename)

    else:
        _logger.info(f'Skipped retrieving "{filename}".')


def _unpack_zip_file(filename, unpack_folder):
    _logger.info(f'Unpacking "{filename}" into "{unpack_folder}".')
    with ZipFile(filename, 'r') as file:
        file.extractall(unpack_folder)


def _download_and_unpack_files(unpack_folder):
    zip_file = 'download/liege.zip'
    url = 'https://www.kaggle.com/api/v1/datasets/download/jonathandumas/liege-microgrid-open-data'
    _download_file(url, filename=zip_file)
    _unpack_zip_file(zip_file, unpack_folder=unpack_folder)


def _compile_pv_dataset(unpack_folder, resample_rate):
    dataset = pd.read_csv(f'{unpack_folder}/miris_pv.csv', index_col='DateTime')

    # Parse timestamps
    dataset.index = pd.to_datetime(dataset.index, utc=True).tz_convert('Europe/Brussels')
    dataset = dataset.resample(resample_rate).mean()

    # Rename columns to follow our convention
    dataset.rename(columns={'PV': 'Target'}, inplace=True)
    dataset.index.name = 'Index'

    return dataset


def _compile_load_dataset(unpack_folder, resample_rate):
    dataset = pd.read_csv(f'{unpack_folder}/miris_load.csv', index_col='DateTime')

    # Parse timestamps
    dataset.index = pd.to_datetime(dataset.index, utc=True).tz_convert('Europe/Brussels')
    dataset = dataset.resample(resample_rate).mean()

    # Rename columns to follow our convention
    dataset.rename(columns={'Conso': 'Target'}, inplace=True)
    dataset.index.name = 'Index'

    return dataset


def _compile_weather_dataset(unpack_folder, resample_rate):
    dataset = pd.read_csv(f'{unpack_folder}/weather_data.csv', index_col='Time')

    # Parse timestamps
    dataset.index = pd.to_datetime(dataset.index, utc=True).tz_convert('Europe/Brussels')

    # TT2M: Temperature two meters above ground
    # Rename columns to follow our convention
    dataset = dataset[['TT2M']]
    dataset.rename(columns={'TT2M': 'AmbientT'}, inplace=True)
    dataset.index.name = 'Index'

    dataset = dataset.resample(resample_rate).mean()

    return dataset


def _split_dataset(dataset, train_ratio):
    dataset = dataset.sort_index()
    split_pos = int(len(dataset) * train_ratio)

    train = dataset[:split_pos]
    test = dataset[split_pos:]
    return train, test


def _save_feature(dataset, save_as):
    float_format = '%.6f'
    dataset.drop(columns=['Target']).to_csv(save_as, float_format=float_format)


def _save_target(dataset, save_as):
    float_format = '%.6f'
    dataset[['Target']].to_csv(save_as, float_format=float_format)


def _compile_dataset_for_da(unpack_folder):
    pv_dataset = _compile_pv_dataset(unpack_folder, resample_rate='1h')
    load_dataset = _compile_load_dataset(unpack_folder, resample_rate='1h')
    weather_dataset = _compile_weather_dataset(unpack_folder, resample_rate='1h')

    pv_dataset = pv_dataset.join(weather_dataset, how='left')
    load_dataset = load_dataset.join(weather_dataset, how='left')

    def _add_features(dataset):
        dataset['Hour'] = dataset.index.hour
        dataset['Month'] = dataset.index.month
        dataset['Day'] = dataset.index.day
        dataset['DoW'] = dataset.index.dayofweek
        dataset['Prior24'] = dataset['Target'].shift(24)
        dataset['Prior48'] = dataset['Target'].shift(48)
        dataset['Prior168'] = dataset['Target'].shift(168)

    _add_features(pv_dataset)
    _add_features(load_dataset)

    pv_dataset = pv_dataset.dropna(axis='index')
    load_dataset = load_dataset.dropna(axis='index')

    pv_train, pv_test = _split_dataset(pv_dataset, train_ratio=0.7)
    load_train, load_test = _split_dataset(load_dataset, train_ratio=0.7)

    _save_feature(pv_train, save_as='da_train/feature-pv.csv')
    _save_target(pv_train, save_as='da_train/target-pv.csv')

    _save_feature(pv_test, save_as='da_test/feature-pv.csv')
    _save_target(pv_test, save_as='da_test/target-pv.csv')

    _save_feature(load_train, save_as='da_train/feature-load.csv')
    _save_target(load_train, save_as='da_train/target-load.csv')

    _save_feature(load_test, save_as='da_test/feature-load.csv')
    _save_target(load_test, save_as='da_test/target-load.csv')


def _compile_dataset_for_id(unpack_folder):
    pv_dataset = _compile_pv_dataset(unpack_folder, resample_rate='15min')
    load_dataset = _compile_load_dataset(unpack_folder, resample_rate='15min')

    pv_dataset = pv_dataset.dropna(axis='index')
    load_dataset = load_dataset.dropna(axis='index')

    pv_train, pv_test = _split_dataset(pv_dataset, train_ratio=0.7)
    load_train, load_test = _split_dataset(load_dataset, train_ratio=0.7)

    _save_target(pv_train, save_as='id_train/target-pv.csv')
    _save_target(pv_test, save_as='id_test/target-pv.csv')
    _save_target(load_train, save_as='id_train/target-load.csv')
    _save_target(load_test, save_as='id_test/target-load.csv')


def _compile_datasets():
    unpack_folder = 'unpack'
    _download_and_unpack_files(unpack_folder)
    _compile_dataset_for_da(unpack_folder)
    _compile_dataset_for_id(unpack_folder)


_compile_datasets()
