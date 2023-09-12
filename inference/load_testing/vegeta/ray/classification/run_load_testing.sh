#!/bin/bash

rps_values=()

for ((rps=1; rps<=5; rps+=1)); do
    rps_values+=($rps)
done

for ((rps=10; rps<=60; rps+=10)); do
    rps_values+=($rps)
done

repetitions=3

for rps in "${rps_values[@]}"; do
    echo "Running load test for $rps RPS..."
    
    for ((i=1; i<=$repetitions; i++)); do
        sudo ./vegeta_script.sh test_text.json $rps $1
        sleep 4
    done

    echo "Load test for $rps RPS completed."
done

echo "All load tests completed."
