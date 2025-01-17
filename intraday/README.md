# Intra-Day Forecasting

```
                    +-----------------------+
                    |  ID Forecast program  | => Prediction file
     Target file => |    (learning mode)    | => Model file
                    +-----------------------+

                    +-----------------------+
     Target file => |  ID Forecast program  | => Prediction file
      Model file => |   (prediction mode)   |
                    +-----------------------+
```

## Forecast Program Usage

To learn a model:

```sh
./id_forecast.py \
    --learn \
    --target ../nyiso_dataset/id_train/target-NYC.csv \
    --model nyiso_results/NYC.model \
    --prediction nyiso_results/train-NYC.csv
```

```sh
./id_forecast.py \
    --learn \
    --target ../liege_dataset/id_train/target-pv.csv \
    --model liege_results/pv.model \
    --prediction liege_results/train-pv.csv
```

Create in-sample predictions. Forecast for interval T is computed using
observations up to T-1.

```sh
./id_forecast.py \
    --in-sample \
    --target ../nyiso_dataset/id_test/target-NYC.csv \
    --model nyiso_results/NYC.model \
    --prediction nyiso_results/test-NYC.csv
```

```sh
./id_forecast.py \
    --in-sample \
    --target ../liege_dataset/id_test/target-pv.csv \
    --model liege_results/pv.model \
    --prediction liege_results/test-pv.csv
```

Create out-of-sample predictions. Forecast for interval T is computed using
previous forecasts.

```sh
./id_forecast.py \
    --target ../nyiso_dataset/id_test/target-NYC.csv \
    --model nyiso_results/NYC.model \
    --prediction nyiso_results/forecast-NYC.csv
```

```sh
./id_forecast.py \
    --target ../liege_dataset/id_test/target-pv.csv \
    --model liege_results/pv.model \
    --prediction liege_results/forecast-pv.csv
```

## Forecast Visualization Program Usage

```sh
./id_visualize.py \
    --prediction nyiso_results/test-NYC.csv \
    --target ../nyiso_dataset/id_test/target-NYC.csv \
    --parse-dates
```

```sh
./id_visualize.py \
    --prediction liege_results/test-pv.csv \
    --target ../liege_dataset/id_test/target-pv.csv \
    --parse-dates
```

## Forecast Validation Program Usage

```sh
./id_validate.py \
    --prediction nyiso_results/test-NYC.csv \
    --target ../nyiso_dataset/id_test/target-NYC.csv
```

```sh
./id_validate.py \
    --prediction liege_results/test-pv.csv \
    --target ../liege_dataset/id_test/target-pv.csv
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
