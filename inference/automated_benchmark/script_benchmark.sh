#!/bin/bash

HUGGINGFACE_REPO="$1"
MODEL_TYPE="$2"
TASK="$3"
RESULT_PATH="$4"
COMPUTE="$5"
SERVER="$6"
INSTANCE_COST="$7"

chmod +x vegeta

chmod +x ./run_load_testing.sh

sudo ./run_load_testing.sh $MODEL_TYPE $TASK $SERVER $RESULT_PATH $HUGGINGFACE_REPO

/opt/conda/bin/python ./process_benchmark_data.py $HUGGINGFACE_REPO $TASK $COMPUTE $SERVER $INSTANCE_COST