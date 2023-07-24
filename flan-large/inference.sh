for i in $(ls prefix-experiments/); do
	python flan_classification_inference.py --adapter_type "prefix-experiments" --experiment ${i} & wait
done
