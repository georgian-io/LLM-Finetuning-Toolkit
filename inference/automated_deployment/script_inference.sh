#!/bin/bash

SERVER="$1"
HUGGINGFACE_REPO="$2"
HUGGINGFACE_TOKEN="$3"
VOLUME=${VOLUME:-"$PWD/data"}

if [ "$SERVER" == "vllm" ]; then
    python3 -m vllm.entrypoints.openai.api_server --model $HUGGINGFACE_REPO

elif [ "$SERVER" == "tgi" ]; then    
    docker run --gpus all --shm-size 1g -e HUGGING_FACE_HUB_TOKEN="$TOKEN" -p 8080:80 -v "$VOLUME":/data ghcr.io/huggingface/text-generation-inference:1.0.3 --model-id "$HUGGINGFACE_REPO"

elif [ "$SERVER" == "ray" ]; then
    serve run servers.ray_serve_vllm:app_builder huggingface_repo=$HUGGINGFACE_REPO

elif [ "$SERVER" == "triton_vllm" ]; then
    json_data="{\"model\":\"$HUGGINGFACE_REPO\",\"disable_log_requests\":\"true\",\"gpu_memory_utilization\":0.9}"
    echo "$json_data" > ./servers/triton_vllm_backend/model_repository/vllm_model/1/model.json
    docker run --gpus all -it --net=host --rm -p 8001:8001 --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/work -w /work nvcr.io/nvidia/tritonserver:23.10-vllm-python-py3 tritonserver --model-store ./servers/triton_vllm_backend/model_repository
fi