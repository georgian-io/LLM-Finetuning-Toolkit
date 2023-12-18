#!/bin/bash

RAW_RESULTS_PATH="$1"
PROCESSED_RESULTS_PATH="$2"
DURATION="$3"
RATE="$4"

chmod +x vegeta

chmod +x ./run_load_testing.sh

./run_load_testing.sh $RAW_RESULTS_PATH $DURATION $RATE

python3 ./process_benchmark_data.py $RAW_RESULTS_PATH $PROCESSED_RESULTS_PATH