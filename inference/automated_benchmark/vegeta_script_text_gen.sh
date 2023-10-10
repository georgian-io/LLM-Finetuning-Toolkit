#!/bin/bash

random_string=$(/opt/conda/bin/python get_prompt.py $1 $2)

echo "{\"inputs\":\"$random_string\"}" > "$3"

./vegeta attack -duration=1s -rate=$4/1s -targets=target.list | ./vegeta report --type=text >> $5