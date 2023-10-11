#!/bin/bash

MODEL_TYPE="$1"
TASK="$2"
RESULT_PATH="$3"

echo $RESULT_PATH

rps_values=()

for ((rps=5; rps<=20; rps+=10)); do
    rps_values+=($rps)
done

repetitions=3

for rps in "${rps_values[@]}"; do
    echo "Running load test for $rps RPS..."
    
    for ((i=1; i<=$repetitions; i++)); do
        random_string=$(/opt/conda/bin/python get_prompt.py $MODEL_TYPE $TASK)
        echo $random_string
        echo "{\"inputs\":\"$random_string\"}" > test_text.json

        ./vegeta attack -duration=1s -rate=$rps/1s -targets=target.list | ./vegeta report --type=text >> $RESULT_PATH
        
        sleep 6
    done

    echo "Load test for $rps RPS completed."
done

echo "All load tests completed."
