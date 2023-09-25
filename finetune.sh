#!/bin/bash

MODEL=$1
TASK=$2
ARGS=${@:3}

echo "Model name: $MODEL"
echo "Task: $TASK"
echo "Arguments: $ARGS"

if [ -z "$MODEL" ]; then
    echo "Please specify the model name."
    exit 1
fi

if [ -z "$TASK" ]; then
    echo "Please specify the task."
    exit 1
fi

if [ -z "$ARGS" ]; then
    echo "Please specify the arguments."
    exit 1
fi

# check that the model exists
if [ ! -d "./$MODEL" ]; then
    echo "Model $MODEL does not exist."
    exit 1
fi

# check that the task exists
if [ ! -f "./$MODEL/${MODEL}_${TASK}.py" ]; then
    echo "Task $TASK does not exist."
    exit 1
fi

cd $MODEL

python "./${MODEL}_${TASK}.py" $ARGS