#!/bin/bash

HUGGINGFACE_REPO="$1"
MODEL_TYPE="$2"
TASK="$3"
RESULT_PATH="$4"
HARDWARE="$5"
SERVER="$6"
INSTANCE_COST="$7"

chmod +x vegeta

chmod +x ./run_load_testing.sh

chmod u+w ./benchmark_results/raw/llama_7b_class/a10/tgi.txt

sudo ./run_load_testing.sh $MODEL_TYPE $TASK $SERVER $RESULT_PATH $HUGGINGFACE_REPO

/opt/conda/bin/python ./process_benchmark_data.py $HUGGINGFACE_REPO $TASK $HARDWARE $SERVER $INSTANCE_COST