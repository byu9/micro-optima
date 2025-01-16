#!/usr/bin/env python3
import logging
from glob import glob
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


def _purge_folder_of_csv_files(folder):
    _logger.info(f'Purging "{folder}".')

    for file in glob(f'{folder}/*.csv'):
        Path(file).unlink()


def _unpack_zip_files(filenames, unpack_folder):
    _purge_folder_of_csv_files(unpack_folder)
    for filename in filenames:
        _unpack_zip_file(filename, unpack_folder)


def _concat_csv_fragments(frag_folder):
    fragments = list()

    for file in glob(f'{frag_folder}/*.csv'):
        _logger.info(f'Reading fragment "{file}".')
        fragments.append(pd.read_csv(file))

    _logger.info('Concatenating fragments.')
    dataset = pd.concat(fragments, axis='index')

    return dataset


def _build_load_dataset(year_month_list):
    fetch_folder = 'fetch_load'
    unpack_folder = 'unpack_load'

    _purge_folder_of_csv_files(unpack_folder)

    for year, month in year_month_list:
        zip_url = f'http://mis.nyiso.com/public/csv/pal/{year:4}{month:02}01pal_csv.zip'
        zip_filename = f'{fetch_folder}/{year}-{month}.zip'
        _download_file(url=zip_url, filename=zip_filename)
        _unpack_zip_file(zip_filename, unpack_folder=unpack_folder)

    dataset = _concat_csv_fragments(unpack_folder).set_index('Time Stamp').sort_index()

    # Localize time stamps
    timezone = 'America/New_York'
    is_edt = dataset.pop('Time Zone') == 'EDT'
    dataset.index = pd.to_datetime(dataset.index).tz_localize(timezone, ambiguous=is_edt)

    return dataset


def _build_weather_dataset(year_month_list):
    fetch_folder = 'fetch_weather'
    unpack_folder = 'unpack_weather'

    _purge_folder_of_csv_files(unpack_folder)

    for year, month in year_month_list:
        zip_url = f'http://mis.nyiso.com/public/csv/lfweather/{year:4}{month:02}01lfweather_csv.zip'
        zip_filename = f'{fetch_folder}/{year}-{month}.zip'
        _download_file(url=zip_url, filename=zip_filename)
        _unpack_zip_file(zip_filename, unpack_folder=unpack_folder)

    dataset = _concat_csv_fragments(unpack_folder)

    # Localize time stamps
    timezone = 'America/New_York'
    dataset['Forecast Date'] = pd.to_datetime(dataset['Forecast Date']).dt.tz_localize(timezone)
    dataset['Vintage Date'] = pd.to_datetime(dataset['Vintage Date']).dt.tz_localize(timezone)

    # Keep only day-ahead weather forecasts
    is_vintage_actual = dataset.pop('Vintage') == 'Actual'
    is_day_ahead = dataset['Forecast Date'] == dataset.pop('Vintage Date') + pd.Timedelta('1d')
    dataset = dataset[is_day_ahead & is_vintage_actual]

    # Index using forecast date
    dataset = dataset.set_index('Forecast Date').sort_index()

    return dataset


def _join_datasets(load, weather):
    # Remove unused load columns
    load = load.reset_index().set_index(['Time Stamp', 'Name'])
    load.drop(columns=['PTID'], inplace=True)

    # Calculate ambient temperature
    # Join Station ID in weather with Station ID in station-to-zone table
    weather = weather.copy()
    weather['AmbientT'] = (weather['Max Temp'] + weather['Min Temp']) / 2
    weather_mapping = pd.read_csv('weather_station_mapping.csv', index_col='Station ID')
    weather = weather.join(weather_mapping, on='Station ID', how='left')

    # Compute mean across all weather stations in the load zone
    # Reindex time stamp to match index of load data
    weather = weather.groupby(['Forecast Date', 'Name'])[['AmbientT']].mean()
    weather = weather.reindex(load.index, method='ffill')

    # Join load with weather, then pivot zone to columns
    _logger.info('Joining load and weather datasets.')
    dataset = load.join(weather, how='left').reset_index().set_index('Time Stamp')
    dataset = pd.pivot(dataset, columns='Name').swaplevel(axis='columns')

    dataset.index.name = 'Index'
    return dataset


def _save_feature(dataset, save_as):
    float_format = '%.6f'
    dataset.drop(columns=['Target']).to_csv(save_as, float_format=float_format)


def _save_target(dataset, save_as):
    float_format = '%.6f'
    dataset[['Target']].to_csv(save_as, float_format=float_format)


def _compile_dataset_for_da(dataset):
    # Resample to hourly datasets
    dataset = dataset.resample('1h').mean()

    # Rename load column as target
    dataset.rename(columns={'Load': 'Target'}, inplace=True)

    def _add_features(_dataset):
        _dataset['Hour'] = _dataset.index.hour
        _dataset['Day'] = _dataset.index.day
        _dataset['Month'] = _dataset.index.month
        _dataset['DoW'] = _dataset.index.dayofweek
        _dataset['Prior24'] = _dataset['Target'].shift(24)
        _dataset['Prior48'] = _dataset['Target'].shift(48)
        _dataset['Prior168'] = _dataset['Target'].shift(168)

    # Split dataset by zones
    for zone in dataset.columns.get_level_values(0):
        zonal = dataset[zone].copy()

        # Construct additional columns
        _add_features(zonal)
        zonal.dropna(axis='index', inplace=True)

        # Split into train and test and drop timestamps
        train_indices = zonal.index < '2024-01-01'
        zonal_train = zonal[train_indices]
        zonal_test = zonal[~train_indices]

        _logger.info(f'Writing datasets for {zone}.')
        _save_feature(zonal_train, f'da_train/feature-{zone}.csv')
        _save_feature(zonal_test, f'da_test/feature-{zone}.csv')
        _save_target(zonal_train, f'da_train/target-{zone}.csv')
        _save_target(zonal_test, f'da_test/target-{zone}.csv')


def _compile_dataset_for_id(dataset):
    # Resample to 15 minute datasets
    dataset = dataset.resample('15min').mean()

    # Rename load column as target
    dataset.rename(columns={'Load': 'Target'}, inplace=True)

    # Split dataset by zones
    for zone in dataset.columns.get_level_values(0):
        zonal = dataset[zone].copy()
        zonal.dropna(axis='index', inplace=True)

        # Split into train and test and drop timestamps
        train_indices = zonal.index < '2024-01-01'
        zonal_train = zonal[train_indices]
        zonal_test = zonal[~train_indices]

        _logger.info(f'Writing datasets for {zone}.')
        _save_target(zonal_train, f'id_train/target-{zone}.csv')
        _save_target(zonal_test, f'id_test/target-{zone}.csv')


def _compile_datasets():
    year_month_list = [
        (year, month)
        for year in [2022, 2023, 2024]
        for month in range(1, 13)
    ]

    load_dataset = _build_load_dataset(year_month_list)
    weather_dataset = _build_weather_dataset(year_month_list)
    dataset = _join_datasets(load_dataset, weather_dataset)

    _compile_dataset_for_da(dataset)
    _compile_dataset_for_id(dataset)


_compile_datasets()
