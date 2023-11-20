#!/bin/bash

RAW_RESULTS_PATH="$1"
PROCESSED_RESULTS_PATH="$2"
HARDWARE="$3"

chmod +x vegeta

chmod +x ./run_load_testing.sh

./run_load_testing.sh $RAW_RESULTS_PATH

python3 ./process_benchmark_data.py $RAW_RESULTS_PATH $PROCESSED_RESULTS_PATH $HARDWARE