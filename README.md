MicroOptima: Energy Management Tool Suite for the Microgrid Project

<!-- TOC -->
  * [Python Dependencies](#python-dependencies)
  * [Dataset Preparation](#dataset-preparation)
  * [Intraday (ID) Forecasting](#intraday-id-forecasting)
  * [Intraday Forecast Program Usage](#intraday-forecast-program-usage)
  * [Intraday Forecast Visualization Program Usage](#intraday-forecast-visualization-program-usage)
  * [Intraday Forecast Validation Program Usage](#intraday-forecast-validation-program-usage)
  * [Intraday Model File Format](#intraday-model-file-format)
  * [Intraday Target File Format](#intraday-target-file-format)
  * [Intraday Prediction File Format](#intraday-prediction-file-format)
  * [Day-Ahead Forecasting](#day-ahead-forecasting)
  * [Day-Ahead Forecast Program Usage](#day-ahead-forecast-program-usage)
  * [Day-Ahead Forecast Visualization Program Usage](#day-ahead-forecast-visualization-program-usage)
  * [Day-Ahead Forecast Validation Program Usage](#day-ahead-forecast-validation-program-usage)
  * [Day-Ahead Feature File Format](#day-ahead-feature-file-format)
  * [Day-Ahead Target File Format](#day-ahead-target-file-format)
  * [Day-Ahead Prediction File Format](#day-ahead-prediction-file-format)
<!-- TOC -->

## Python Dependencies

Python interpreter version 3.12 is recommended. Run the following command to
install all dependencies.

```sh
pip install -r requirements.txt
```

## Dataset Preparation

Run the following commands

```sh
$(cd datasets/nyiso_dataset; ./compile_dataset.py)
$(cd datasets/liege_dataset; ./compile_dataset.py)
```

## Intraday (ID) Forecasting

```
                    +-----------------------+
                    |  ID Forecast program  | => ID Prediction file
  ID Target file => |    (learning mode)    | => ID Model file
                    +-----------------------+

                    +-----------------------+
  ID Target file => |  ID Forecast program  | => ID Prediction file
   ID Model file => |   (prediction mode)   |
                    +-----------------------+
```

The following models are currently supported.

| Type     | Description                                                                        |
|----------|------------------------------------------------------------------------------------|
| `ar8`    | 8th order linear auto-regressor                                                    |
| `dt8`    | 8th order decision tree auto-regressor                                             |
| `rf8`    | 8th order random forest auto-regressor                                             |
| `nn8k20` | 8th order 20 nearest neighbors auto-regressor                                      |
| `mlp111` | 8th order multi-layer perceptron auto-regressor with (100, 100, 100) hidden layers |
| `zoh`    | zero order hold                                                                    |

## Intraday Forecast Program Usage

Run the program in learning mode to create a forecast model and write the
predictions on the training dataset to file.

```sh
./id_forecast.py \
    --learn=ar8 \
    --target datasets/id_train/target-liege-PV.csv \
    --model results/id/target-liege-PV.model \
    --prediction results/id/train-liege-PV.csv
```

Run the program in prediction mode by loading the forecast model and write the
predictions on the test dataset to file.

```sh
./id_forecast.py \
    --target datasets/id_test/target-liege-PV.csv \
    --model results/id/target-liege-PV.model \
    --prediction results/id/test-liege-PV.csv
```

If unsure, run the program with `--help` to display help. The following is
sample output from a previous version. The actual message displayed may differ
in a subsequent version.

```
./id_forecast.py --help
usage: id_forecast.py [-h] [--learn {ar8,dt8,nn8k20,rf8,mlp111}] --model MODEL --target TARGET
                      [--prediction PREDICTION]

Learns an intraday forecast model or uses a learned model to make a forecast.

options:
  -h, --help            show this help message and exit
  --learn {ar8,dt8,nn8k20,rf8,mlp111}
                        put the program in learn mode and learn with specified model
  --model MODEL         in learn mode, the file path to write the model to; in forecast mode,
                        the file path to load the model from
  --target TARGET       the file path to load historical observations from
  --prediction PREDICTION
                        the file path to write predictions to
```

## Intraday Forecast Visualization Program Usage

```sh
./id_visualize.py \
    --prediction results/id/test-liege-PV.csv \
    --target datasets/id_test/target-liege-PV.csv \
    --parse-dates \
    --title liege-PV
```

If unsure, run the program with `--help` to display help. The following is
sample output from a previous version. The actual message displayed may differ
in a subsequent version.

```
./id_visualize.py --help
usage: id_visualize.py [-h] --prediction PREDICTION --target TARGET
                       [--parse-dates | --no-parse-dates] [--title TITLE] [--save SAVE]

Visualizes the prediction of a intraday forecast model.

options:
  -h, --help            show this help message and exit
  --prediction PREDICTION
                        path to load the prediction file from
  --target TARGET       path to load the target file from
  --parse-dates, --no-parse-dates
                        parse the index column as timestamps instead of observation numbers
  --title TITLE         use the given title in the plot
  --save SAVE           save the plot to the given path as image (image type determined from
                        suffix)
```

## Intraday Forecast Validation Program Usage

```sh
./id_validate.py \
    --pairs id-pairs-liege.csv \
    --scoreboard scoreboard-liege.csv 
```

If unsure, run the program with `--help` to display help. The following is
sample output from a previous version. The actual message displayed may differ
in a subsequent version.

```
./id_validate.py --help
usage: id_validate.py [-h] --pairs PAIRS --scoreboard SCOREBOARD

Validates intraday predictions against actual observations.

options:
  -h, --help            show this help message and exit
  --pairs PAIRS         CSV file listing prediction/target file pairs
  --scoreboard SCOREBOARD
                        file to write the validation scores to
```

## Intraday Model File Format

The model file is a pickled python object.

## Intraday Target File Format

In prediction mode, the number of observations in the target file must be at
least the order of the model. For example, for `ar8` (8th order linear
auto-regressor), the target file must contain at least 8 observations to make a
prediction.

| Column | Range | Description                                      |
|--------|-------|--------------------------------------------------|
| Index  | Any   | Observation key.                                 |
| Target | Any   | Prediction target over the observation interval. |

## Intraday Prediction File Format

| Column | Range | Description                                    |
|--------|-------|------------------------------------------------|
| Index  | Any   | Observation key.                               |
| Pred0  | Any   | Prediction for the first prediction interval.  |
| Pred1  | Any   | Prediction for the second prediction interval. |
| Pred2  | Any   | Prediction for the third prediction interval.  |
| Pred3  | Any   | Prediction for the fourth prediction interval. |

## Day-Ahead Forecasting

```
                    +-----------------------+
 DA Feature file => |  DA Forecast program  | => DA Prediction file
  DA Target file => |    (learning mode)    | => DA Model file
                    +-----------------------+

                    +-----------------------+
 DA Feature file => |  DA Forecast program  | => DA Prediction file
  DA  Model file => |   (prediction mode)   |
                    +-----------------------+
```

## Day-Ahead Forecast Program Usage

Run the program in learning mode to create a forecast model and write the
predictions on the training dataset to file.

```sh
  ./da_forecast.py \
    --learn=fuzzyprob \
    --feature datasets/da_train/feature-liege-PV.csv \
    --target datasets/da_train/target-liege-PV.csv \
    --model results/da/target-liege-PV.model \
    --prediction results/da/train-liege-PV.csv
```

Run the program in prediction mode by loading the forecast model and write the
predictions on the test dataset to file.

```sh
  ./da_forecast.py \
    --feature datasets/da_test/feature-liege-PV.csv \
    --target datasets/da_test/target-liege-PV.csv \
    --model results/da/target-liege-PV.model \
    --prediction results/da/test-liege-PV.csv
```

If unsure, run the program with `--help` to display help. The following is
sample
output from a previous version. The actual message displayed may differ in a
subsequent version.

```
./da_forecast.py --help
usage: da_forecast.py [-h] [--learn {fuzzyprob,gpr}] --model MODEL --feature FEATURE --target
                      TARGET [--prediction PREDICTION]

Learns a day-ahead forecast model or uses a learned model to make a forecast.

options:
  -h, --help            show this help message and exit
  --learn {fuzzyprob,gpr}
                        put the program in learn mode and learn with specified model
  --model MODEL         in learn mode, the file path to write the model to; in forecast mode,
                        the file path to load the model from
  --feature FEATURE     the file path to load historical covariates from
  --target TARGET       the file path to load historical observations from
  --prediction PREDICTION
                        the file path to write predictions to
```

## Day-Ahead Forecast Visualization Program Usage

```sh
  ./da_visualize.py \
    --prediction results/da/test-liege-PV.csv \
    --target datasets/da_test/target-liege-PV.csv \
    --parse-dates \
    --title liege-PV 
```

If unsure, run the program with `--help` to display help. The following is
sample
output from a previous version. The actual message displayed may differ in a
subsequent version.

```
./da_visualize.py --help
usage: da_visualize.py [-h] --prediction PREDICTION --target TARGET
                       [--parse-dates | --no-parse-dates] [--title TITLE] [--save SAVE]

Visualizes the prediction of a intraday forecast model.

options:
  -h, --help            show this help message and exit
  --prediction PREDICTION
                        path to load the prediction file from
  --target TARGET       path to load the target file from
  --parse-dates, --no-parse-dates
                        parse the index column as timestamps instead of observation numbers
  --title TITLE         use the given title in the plot
  --save SAVE           save the plot to the given path as image (image type determined from
                        suffix)
```

## Day-Ahead Forecast Validation Program Usage

```sh
  ./da_validate.py \
    --pairs da-pairs-nyiso.csv \
    --scoreboard results/da/validate-nyiso.csv
```

If unsure, run the program with `--help` to display help. The following is
sample output from a previous version. The actual message displayed may differ
in a subsequent version.

```
./da_validate.py --help
usage: da_validate.py [-h] --pairs PAIRS --scoreboard SCOREBOARD

Validates intraday predictions against actual observations.

options:
  -h, --help            show this help message and exit
  --pairs PAIRS         CSV file listing prediction/target file pairs
  --scoreboard SCOREBOARD
                        file to write the validation scores to
```

## Day-Ahead Feature File Format

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

## Day-Ahead Target File Format

| Column | Range | Description                                 |
|--------|-------|---------------------------------------------|
| Index  | Any   | Observation key.                            |
| Target | Any   | Prediction target at the forecast interval. |

## Day-Ahead Prediction File Format

| Column | Range        | Description                                                      |
|--------|--------------|------------------------------------------------------------------|
| Index  | Any          | Observation key.                                                 |
| Mean   | Any          | Predicted mean of target at the forecast interval.               |
| Std    | Non-negative | Predicted standard deviation of target at the forecast interval. |
