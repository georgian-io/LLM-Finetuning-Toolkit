#!/bin/bash

MODEL_TYPE="$1"
TASK="$2"
INPUT_FILE="$3"
RESULT_PATH="$4"

random_string=$(/opt/conda/bin/python get_prompt.py $MODEL_TYPE $TASK)

echo "{\"inputs\":\"$random_string\"}" > "$INPUT_FILE"

./vegeta attack -duration=1s -rate=$4/1s -targets=target.list | ./vegeta report --type=text >> $RESULT_PATH