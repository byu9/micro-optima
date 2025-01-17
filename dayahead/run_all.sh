#!/bin/sh

train_nyiso_model() {
  echo train_nyiso_model $1
  ./da_forecast.py \
    --learn \
    --feature ../nyiso_dataset/da_train/feature-$1.csv \
    --target ../nyiso_dataset/da_train/target-$1.csv \
    --model nyiso_results/$1.model \
    --prediction nyiso_results/train-$1.csv
}

forecast_nyiso_model() {
  echo forecast_nyiso_model $1
  ./da_forecast.py \
    --feature ../nyiso_dataset/da_test/feature-$1.csv \
    --model nyiso_results/$1.model \
    --prediction nyiso_results/test-$1.csv
}

validate_nyiso_model() {
  ./da_validate.py \
    --prediction nyiso_results/test-$1.csv \
    --target ../nyiso_dataset/da_test/target-$1.csv
}

train_liege_model() {
  echo train_liege_model $1
  ./da_forecast.py \
    --learn \
    --feature ../liege_dataset/da_train/feature-$1.csv \
    --target ../liege_dataset/da_train/target-$1.csv \
    --model liege_results/$1.model \
    --prediction liege_results/train-$1.csv
}

forecast_liege_model() {
  echo forecast_nyiso_model $1
  ./da_forecast.py \
    --feature ../liege_dataset/da_test/feature-$1.csv \
    --model liege_results/$1.model \
    --prediction liege_results/test-$1.csv
}

validate_liege_model() {
  ./da_validate.py \
    --prediction liege_results/test-$1.csv \
    --target ../liege_dataset/da_test/target-$1.csv
}

run_task() {
  zones_var=$1_zones

  > ${1}_results.txt
  for zone in ${!zones_var}
  do
    train_${1}_model $zone
    forecast_${1}_model $zone
    echo $zone $(validate_${1}_model $zone) >> ${1}_results.txt
  done
}

liege_zones='pv load'
nyiso_zones='
CAPITL CENTRL DUNWOD GENESE HUD_VL LONGIL
MHK_VL MILLWD NORTH NYC WEST
'

run_task liege
run_task nyiso
