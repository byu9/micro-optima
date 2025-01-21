#!/bin/sh

timestamp=$(date '+%FT%T')

train_model() {
  echo train model $1
  ./id_forecast.py \
    --learn=$2 \
    --target datasets/id_train/target-$1.csv \
    --model results/id/target-$1.model \
    --prediction results/id/train-$1.csv
}

test_model() {
  echo test model $1
  ./id_forecast.py \
    --target datasets/id_test/target-$1.csv \
    --model results/id/target-$1.model \
    --prediction results/id/test-$1.csv
}

visualize_prediction() {
  ./id_visualize.py \
    --prediction results/id/test-$1.csv \
    --target datasets/id_test/target-$1.csv \
    --parse-dates \
    --title $zone \
    --save results/id/test-$1.png
}

validate_model() {
  ./id_validate.py \
    --pairs id-pairs-$1.csv \
    --scoreboard results/id/validate-$1-$timestamp.csv
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

  validate_model nyiso $1
  validate_model liege $1
}

run_task zoh
