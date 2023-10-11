#!/bin/bash

SCRIPT_PATH="$1"
MODEL_TYPE="$2"
TASK="$3"
RESULT_PATH="$4"

rps_values=()

for ((rps=5; rps<=190; rps+=10)); do
    rps_values+=($rps)
done

repetitions=3

for rps in "${rps_values[@]}"; do
    echo "Running load test for $rps RPS..."
    
    for ((i=1; i<=$repetitions; i++)); do
        sudo $SCRIPT_PATH $MODEL_TYPE $TASK test_text.json $rps $RESULT_PATH
        sleep 4
    done

    echo "Load test for $rps RPS completed."
done

echo "All load tests completed."
