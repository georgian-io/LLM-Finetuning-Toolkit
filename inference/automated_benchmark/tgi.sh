#!/bin/bash

HUGGINGFACE_REPO="$1"
VOLUME=${VOLUME:-"$PWD/data"}
TOKEN="$2"
MODEL_TYPE="$3"
TASK="$4"
RESULT_PATH="$5"
COMPUTE="$6"
SERVER="$7"
INSTANCE_COST="$8"

docker run --gpus all --shm-size 1g -e HUGGING_FACE_HUB_TOKEN="$TOKEN" -p 8080:80 -v "$VOLUME":/data ghcr.io/huggingface/text-generation-inference:1.0.3 --model-id "$HUGGINGFACE_REPO"

sleep 300
#while ! python -c "import sys, requests; sys.exit(requests.post('http://0.0.0.0:8080/generate', json={'inputs': 'hahaha'}).status_code != 200)"; do
#  echo "Waiting for server to start..."
#  sleep 60
#done

python -c "import sys, requests; sys.exit(requests.post('http://0.0.0.0:8080/generate', json={'inputs': 'hahaha'}).status_code != 200)"

echo "Server has started, running benchmark..."

chmod +x vegeta

chmod +x ./run_load_testing.sh

echo $RESULT_PATH

#./run_load_testing.sh $MODEL_TYPE $TASK $RESULT_PATH

/opt/conda/bin/python ./benchmark_results/process_benchmark.py $HUGGINGFACE_REPO $TASK $COMPUTE $SERVER $INSTANCE_COST

#model_name, task, compute, server, instance_cost