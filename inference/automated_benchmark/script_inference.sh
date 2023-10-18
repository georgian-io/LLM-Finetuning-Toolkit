#!/bin/bash

HUGGINGFACE_REPO="$1"
TOKEN="$2"
MODEL_TYPE="$3"
TASK="$4"
SERVER="$5"
VOLUME=${VOLUME:-"$PWD/data"}
PATH_TO_LORA_WEIGHTS="$6"

if [ "$SERVER" == "vllm" ]; then
    python3 -m vllm.entrypoints.openai.api_server --model $HUGGINGFACE_REPO
elif [ "$SERVER" == "tgi" ]; then    
    docker run --gpus all --shm-size 1g -e HUGGING_FACE_HUB_TOKEN="$TOKEN" -p 8080:80 -v "$VOLUME":/data ghcr.io/huggingface/text-generation-inference:1.0.3 --model-id "$HUGGINGFACE_REPO"
elif [ "$SERVER" == "ray" ]; then
    serve run ray_serve:app_builder model_type=$MODEL_TYPE task=$TASK path_to_lora_weights=$PATH_TO_LORA_WEIGHTS
fi