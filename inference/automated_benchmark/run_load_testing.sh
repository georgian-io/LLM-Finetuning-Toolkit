#!/bin/bash

rps_values=()

for ((rps=5; rps<=190; rps+=10)); do
    rps_values+=($rps)
done

repetitions=3

for rps in "${rps_values[@]}"; do
    echo "Running load test for $rps RPS..."
    
    for ((i=1; i<=$repetitions; i++)); do
        sudo $1 $2 $3 test_text.json $rps $4
        sleep 4
    done

    echo "Load test for $rps RPS completed."
done

echo "All load tests completed."
