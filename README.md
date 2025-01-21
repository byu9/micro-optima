# MicroOptima

Energy Management Tool Suite for Microgrid C³ Project

## Python Dependencies

Python interpreter version 3.12 is recommended. Run the following command to
install all dependencies.

```commandline
pip install -r requirements.txt
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

## Intraday Forecast Program Usage

Run the program in learning mode to create a forecast model and write the
predictions on the training dataset to file.

```sh
./id_forecast.py \
    --learn=ar8 \
    --target datasets/id_train/target-nyiso-NYC.csv \
    --model results/id/target-nyiso-NYC.model \
    --prediction results/id/train-nyiso-NYC.csv
```

Run the program in prediction mode by loading the forecast model and write the
predictions on the test dataset to file.

```sh
./id_forecast.py \
    --target datasets/id_test/target-nyiso-NYC.csv \
    --model results/id/target-nyiso-NYC.model \
    --prediction results/id/test-nyiso-NYC.csv
```

If unsure, run the program with `--help` to display help. The following is
sample output from a previous version. The actual message displayed may differ
in a subsequent version.

```
./id_forecast.py --help
usage: id_forecast.py [-h] [--learn {ar8,dt8,nn8k20,rf8,mlp111}] --model MODEL --target TARGET
                      [--prediction PREDICTION]

Learns a intra-day forecast model or use a learned model to make a forecast.

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
    --prediction results/id/test-nyiso-NYC.csv \
    --target datasets/id_test/target-nyiso-NYC.csv \
    --parse-dates \
    --title nyiso-NYC \
    --save results/id/test-nyiso-NYC.png
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
