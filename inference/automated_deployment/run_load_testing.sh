#!/bin/bash

RESULTS_PATH="$1"
DURATION="$2"
RATE=$(("$3"))

repetitions=2

for ((i=1; i<=$repetitions; i++)); do
    request=$(python3 send_post_request.py benchmark)
    echo "$request" > input.json

    ./vegeta attack -duration=$DURATION -rate=$RATE/1s -targets=target.list | ./vegeta report --type=text >> "${RESULTS_PATH}"
    
    sleep 5
done

 
