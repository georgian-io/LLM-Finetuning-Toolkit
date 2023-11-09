#!/bin/bash

HUGGINGFACE_REPO="$1"
TOKEN="$2"
MODEL_TYPE="$3"
TASK="$4"
SERVER="$5"
VOLUME=${VOLUME:-"$PWD/data"}
PATH_TO_LORA_WEIGHTS="$6"
#AWS_ROLE="$7"
#AWS_ACCESS_KEY_ID="$8"
#AWS_SECRET_ACCESS_KEY="$9"
#AWS_SESSION_TOKEN="$10"

if [ "$SERVER" == "vllm" ]; then
    python3 -m vllm.entrypoints.openai.api_server --model $HUGGINGFACE_REPO
elif [ "$SERVER" == "tgi" ]; then    
    docker run --gpus all --shm-size 1g -e HUGGING_FACE_HUB_TOKEN="$TOKEN" -p 8080:80 -v "$VOLUME":/data ghcr.io/huggingface/text-generation-inference:1.0.3 --model-id "$HUGGINGFACE_REPO"
elif [ "$SERVER" == "ray" ]; then
    serve run ray_serve:app_builder model_type=$MODEL_TYPE task=$TASK path_to_lora_weights=$PATH_TO_LORA_WEIGHTS
elif [ "$SERVER" == "triton_vllm" ]; then
    json_data="{\"model\":\"$HUGGINGFACE_REPO\",\"disable_log_requests\":\"true\",\"gpu_memory_utilization\":0.9}"
    echo "$json_data" > ./servers/triton_vllm_backend/model_repository/vllm_model/1/model.json
    docker run --gpus all -it --net=host --rm -p 8001:8001 --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/work -w /work nvcr.io/nvidia/tritonserver:23.10-vllm-python-py3 tritonserver --model-store ./servers/triton_vllm_backend/model_repository
fi