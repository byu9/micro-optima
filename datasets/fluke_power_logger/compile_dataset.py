#!/usr/bin/env python3

import pandas as pd


def _read_logger_a():
    dataframe = pd.read_excel('Logger A/Logger_A_1.xlsx', usecols=[1, 5], names=['Index', 'Target'])
    dataframe['Index'] = pd.to_datetime(dataframe['Index'])
    dataframe.set_index('Index', inplace=True)
    return dataframe


def _read_logger_b():
    dataframe = pd.read_excel('Logger B/Logger_B_1.xlsx', usecols=[1, 11], names=['Index', 'Target'])
    dataframe['Index'] = pd.to_datetime(dataframe['Index'])
    dataframe.set_index('Index', inplace=True)
    return dataframe


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


def _compile_dataset_for_da(dataframe):
    dataframe = dataframe.resample('1h').mean()

    dataframe['Hour'] = dataframe.index.hour
    dataframe['Month'] = dataframe.index.month
    dataframe['Day'] = dataframe.index.day
    dataframe['DoW'] = dataframe.index.dayofweek
    dataframe['Prior24'] = dataframe['Target'].shift(24)
    dataframe['Prior48'] = dataframe['Target'].shift(48)
    dataframe['Prior168'] = dataframe['Target'].shift(168)

    dataframe.dropna(axis='index', inplace=True)
    return dataframe


def _compile_dataset_for_id(dataframe):
    dataframe = dataframe.resample('15min').mean()
    dataframe.ffill(inplace=True)
    return dataframe


def _compile_datasets_fluke_a():
    dataset = _read_logger_a()
    da_dataset = _compile_dataset_for_da(dataset)
    id_dataset = _compile_dataset_for_id(dataset)

    da_train, da_test = _split_dataset(da_dataset, train_ratio=0.7)
    id_train, id_test = _split_dataset(id_dataset, train_ratio=0.7)

    _save_feature(da_train, save_as='../da_train/feature-fluke-a.csv')
    _save_target(da_train, '../da_train/target-fluke-a.csv')

    _save_feature(da_test, '../da_test/feature-fluke-a.csv')
    _save_target(da_test, '../da_test/target-fluke-a.csv')

    _save_target(id_train, '../id_train/target-fluke-a.csv')
    _save_target(id_test, '../id_test/target-fluke-a.csv')


def _compile_datasets_fluke_b():
    dataset = _read_logger_b()
    da_dataset = _compile_dataset_for_da(dataset)
    id_dataset = _compile_dataset_for_id(dataset)

    da_train, da_test = _split_dataset(da_dataset, train_ratio=0.7)
    id_train, id_test = _split_dataset(id_dataset, train_ratio=0.7)

    _save_feature(da_train, save_as='../da_train/feature-fluke-b.csv')
    _save_target(da_train, '../da_train/target-fluke-b.csv')

    _save_feature(da_test, '../da_test/feature-fluke-b.csv')
    _save_target(da_test, '../da_test/target-fluke-b.csv')

    _save_target(id_train, '../id_train/target-fluke-b.csv')
    _save_target(id_test, '../id_test/target-fluke-b.csv')


def _compile_datasets():
    _compile_datasets_fluke_a()
    _compile_datasets_fluke_b()


_compile_datasets()
