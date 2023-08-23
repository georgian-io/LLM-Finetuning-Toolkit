#!/bin/bash

rps_values=()

for ((rps=10; rps<=300; rps+=15)); do
    rps_values+=($rps)
done

repetitions=3

for rps in "${rps_values[@]}"; do
    echo "Running load test for $rps RPS..."
    
    for ((i=1; i<=$repetitions; i++)); do
        sudo $1 test_text.json $rps $2
        sleep 4
    done

    echo "Load test for $rps RPS completed."
done

echo "All load tests completed."