#!/bin/sh

timestamp=$(date '+%FT%T')

train_model() {
  echo train model $1
  ./da_forecast.py \
    --learn=$2\
    --feature datasets/da_train/feature-$1.csv \
    --target datasets/da_train/target-$1.csv \
    --model results/da/target-$1.model \
    --prediction results/da/train-$1.csv
}

test_model() {
  echo test model $1
  ./da_forecast.py \
    --feature datasets/da_test/feature-$1.csv \
    --target datasets/da_test/target-$1.csv \
    --model results/da/target-$1.model \
    --prediction results/da/test-$1.csv
}

visualize_prediction() {
  ./da_visualize.py \
    --prediction results/da/test-$1.csv \
    --target datasets/da_test/target-$1.csv \
    --parse-dates \
    --title $zone \
    --save results/da/test-$1.pdf
}

validate_model() {
  ./da_validate.py \
    --pairs da-pairs-$1.csv \
    --scoreboard results/da/validate-$1-$timestamp.csv
}

zones='
nyiso-CAPITL nyiso-CENTRL nyiso-DUNWOD nyiso-GENESE nyiso-HUD_VL
nyiso-LONGIL nyiso-MHK_VL nyiso-MILLWD nyiso-NORTH nyiso-NYC
nyiso-WEST liege-load liege-pv
'

run_task() {
  for zone in $zones
  do
    train_model $zone $1
    test_model $zone
    visualize_prediction $zone
  done

  validate_model nyiso
  validate_model liege
}

run_task xgboost
