#!/bin/bash

json_file="$1"

SERVER=$(jq -r '.server' "$json_file")

if [ "$SERVER" != "ray" ]; then
    HUGGINGFACE_REPO=$(jq -r '.huggingface_repo' "$json_file")
fi

if [ "$SERVER" == "vllm" ]; then
    python3 -m vllm.entrypoints.openai.api_server --model $HUGGINGFACE_REPO

elif [ "$SERVER" == "tgi" ]; then    
    HUGGINGFACE_TOKEN=$(jq -r '.huggingface_token' "$json_file")
    VOLUME=${VOLUME:-"$PWD/data"}
    docker run --gpus all --shm-size 1g -e HUGGING_FACE_HUB_TOKEN="$TOKEN" -p 8080:80 -v "$VOLUME":/data ghcr.io/huggingface/text-generation-inference:1.0.3 --model-id "$HUGGINGFACE_REPO"

elif [ "$SERVER" == "ray" ]; then
    MODEL_TYPE=$(jq -r '.model_type' "$json_file")
    TASK=$(jq -r '.task' "$json_file")
    PATH_TO_LORA_WEIGHTS=$(jq -r '.path_to_model' "$json_file")
    serve run servers.ray_serve:app_builder model_type=$MODEL_TYPE task=$TASK path_to_lora_weights=$PATH_TO_LORA_WEIGHTS

elif [ "$SERVER" == "triton_vllm" ]; then
    json_data="{\"model\":\"$HUGGINGFACE_REPO\",\"disable_log_requests\":\"true\",\"gpu_memory_utilization\":0.9}"
    echo "$json_data" > ./servers/triton_vllm_backend/model_repository/vllm_model/1/model.json
    docker run --gpus all -it --net=host --rm -p 8001:8001 --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/work -w /work nvcr.io/nvidia/tritonserver:23.10-vllm-python-py3 tritonserver --model-store ./servers/triton_vllm_backend/model_repository
fi