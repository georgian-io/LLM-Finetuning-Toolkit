#!/bin/bash

HUGGINGFACE_REPO="$1"
MODEL_TYPE="$3"
TASK="$4"
COMPUTE="$5"
SERVER="$6"
VOLUME=${VOLUME:-"$PWD/data"}
LORA_WEIGHTS="$7"

if [ "$SERVER" == "vllm" ]; then
    python3 -m vllm.entrypoints.openai.api_server --model $HUGGINGFACE_REPO
elif [ "$SERVER" == "tgi" ]; then
    echo "Hi!"    
    docker run --gpus all --shm-size 1g -e HUGGING_FACE_HUB_TOKEN="$TOKEN" -p 8080:80 -v "$VOLUME":/data ghcr.io/huggingface/text-generation-inference:1.0.3 --model-id "$HUGGINGFACE_REPO"
elif [ "$SERVER" == "ray" ]; then
    serve run ray_serve:app_builder model_type=$MODEL_TYPE task=$TASK lora_weights=$LORA_WEIGHTS
fi

