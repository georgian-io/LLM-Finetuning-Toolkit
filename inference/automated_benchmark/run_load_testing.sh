#!/bin/bash

MODEL_TYPE="$1"
TASK="$2"
SERVER="$3"
RESULT_PATH="$4"
HUGGINGFACE_REPO="$5"

rps_values=(10)

#for ((rps=5; rps<=240; rps+=10)); do
#    rps_values+=($rps)
#done

repetitions=3

for rps in "${rps_values[@]}"; do
    echo "Running load test fsor $rps RPS..."
    
    for ((i=1; i<=$repetitions; i++)); do
        request=$(/opt/conda/bin/python send_post_request.py $MODEL_TYPE $TASK $SERVER $HUGGINGFACE_REPO)

        echo "$request" > input.json

        ./vegeta attack -duration=600s -rate=$rps/1s -targets=target.list | ./vegeta report --type=text >> $RESULT_PATH
        
        sleep 3
    done

    echo "Load test for $rps RPS completed."
done

echo "All load tests completed."
