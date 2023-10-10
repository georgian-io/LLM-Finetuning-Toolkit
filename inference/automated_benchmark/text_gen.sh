#!/bin/bash

HUGGINGFACE_REPO="$1"
VOLUME=${VOLUME:-"$PWD/data"}
TOKEN="$2"

docker run -d --gpus all --shm-size 1g -e HUGGING_FACE_HUB_TOKEN="$TOKEN" -p 8080:80 -v "$VOLUME":/data ghcr.io/huggingface/text-generation-inference:1.0.3 --model-id "$HUGGINGFACE_REPO"

while ! python -c "import sys, requests; sys.exit(requests.post('http://0.0.0.0:8080/generate', json={'inputs': 'hahaha'}).status_code != 200)"; do
  echo "Waiting for server to start..."
  sleep 2
done

echo "Server has started, running benchmark..."

chmod +x vegeta

chmod +x ./vegeta_scripts/vegeta_script_text_gen.sh

chmod +x ./run_load_testing.sh

./run_load_testing.sh ./vegeta_scripts/vegeta_script_text_gen.sh $3 $4 $5

/opt/conda/bin/python ./benchmark_results/process_benchmark.py 