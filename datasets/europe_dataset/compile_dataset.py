#!/usr/bin/env python3
import logging

import pandas as pd

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _save_feature(dataset, save_as):
    float_format = '%.6f'
    dataset.drop(columns=['Target']).to_csv(save_as, float_format=float_format, index_label='Index')


def _save_target(dataset, save_as):
    float_format = '%.6f'
    dataset[['Target']].to_csv(save_as, float_format=float_format, index_label='Index')


def _compile_dataset_for_da(country):
    dataset = pd.read_csv(f'raw/{country}.csv', index_col='utc')

    # Rename load column as target
    dataset.rename(columns={'mw': 'Target'}, inplace=True)

    # Split into train and test and drop timestamps
    split_loc = -1465
    train = dataset[:split_loc]
    test = dataset[split_loc:]

    _logger.info(f'Writing datasets for {country}.')
    _save_feature(train, f'../da_train/feature-europe-{country}.csv')
    _save_feature(test, f'../da_test/feature-europe-{country}.csv')
    _save_target(train, f'../da_train/target-europe-{country}.csv')
    _save_target(test, f'../da_test/target-europe-{country}.csv')


def _compile_datasets():
    country_list = [
        'AT', 'BE', 'BG', 'CH', 'CZ', 'DK', 'ES', 'FR', 'GR', 'IT', 'NL', 'PT', 'SI', 'SK'
    ]
    for country in country_list:
        _compile_dataset_for_da(country)


_compile_datasets()
