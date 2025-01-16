# Day-Ahead Forecasting

```
                    +--------------------+
    Feature file => |  Forecast program  | => Prediction file
     Target file => |  (learning mode)   | => Model file
                    +--------------------+

                    +--------------------+
    Feature file => |  Forecast program  | => Prediction file
      Model file => |  (prediction mode) |
                    +--------------------+
```

## Forecast Program Usage

To learn a model:

```sh
./da_forecast.py \
    --learn \
    --feature ../nyiso_dataset/da_train/feature-N.Y.C..csv \
    --target ../nyiso_dataset/da_train/target-N.Y.C..csv \
    --model nyiso_results/N.Y.C..model \
    --prediction nyiso_results/train-N.Y.C..csv
```

```sh
./da_forecast.py \
    --learn \
    --feature ../liege_dataset/da_train/feature-pv.csv \
    --target ../liege_dataset/da_train/target-pv.csv \
    --model liege_results/pv.model \
    --prediction liege_results/train-pv.csv
```

To use the learned model to forecast

```sh
./da_forecast.py \
    --feature ../nyiso_dataset/da_test/feature-N.Y.C..csv \
    --model nyiso_results/N.Y.C..model \
    --prediction nyiso_results/test-N.Y.C..csv
```

```sh
./da_forecast.py \
    --feature ../liege_dataset/da_test/feature-pv.csv \
    --model liege_results/pv.model \
    --prediction liege_results/test-pv.csv
```

## Forecast Visualization Program Usage

```sh
./da_visualize.py \
    --prediction nyiso_results/test-N.Y.C..csv \
    --target ../nyiso_dataset/da_test/target-N.Y.C..csv \
    --parse-dates
```

```sh
./da_visualize.py \
    --prediction liege_results/test-pv.csv \
    --target ../liege_dataset/da_test/target-pv.csv \
    --parse-dates
```

## Model file format

The model file is a pickled python object.

## Feature file format

| Column   | Range | Description                                                                                    |
|----------|-------|------------------------------------------------------------------------------------------------|
| Index    | Any   | Observation key.                                                                               |
| Month    | 1-12  | Month (local time) corresponding to the forecast interval.                                     |
| Day      | 1-31  | Day (local time) corresponding to the forecast interval.                                       |
| Hour     | 0-23  | Hour of day (local time) corresponding to the forecast interval.                               |
| DoW      | 0-6   | Day of week (local time) corresponding to the forecast interval.                               |
| AmbientT | Any   | Predicted ambient temperature in Celsius or Fahrenheit corresponding to the forecast interval. |
| Prior24  | Any   | Prediction target over the interval that is 24 hours earlier than the forecast interval.       |
| Prior48  | Any   | Prediction target over the interval that is 48 hours earlier than the forecast interval.       |
| Prior168 | Any   | Prediction target over the interval that is 168 hours earlier than the forecast interval.      |

## Target file format

| Column | Range | Description                                 |
|--------|-------|---------------------------------------------|
| Index  | Any   | Observation key.                            |
| Target | Any   | Prediction target at the forecast interval. |

## Prediction file format

| Column | Range        | Description                                                      |
|--------|--------------|------------------------------------------------------------------|
| Index  | Any          | Observation key.                                                 |
| Mean   | Any          | Predicted mean of target at the forecast interval.               |
| Std    | Non-negative | Predicted standard deviation of target at the forecast interval. |
