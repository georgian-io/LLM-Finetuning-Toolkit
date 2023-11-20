#!/bin/bash

RESULTS_PATH="$1"
rps=10

repetitions=1

for ((i=1; i<=$repetitions; i++)); do
    request=$(python3 send_post_request.py)

    echo "$request" > input.json

    ./vegeta attack -duration=5s -rate=$rps/1s -targets=target.list | ./vegeta report --type=text >> "${RESULTS_PATH}"
    
    sleep 5
done

 
