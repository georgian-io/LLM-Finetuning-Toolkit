sample_fraction=(0.025 0.05 0.1)

for (( sf=0; sf<3; sf=sf+1 )) do
	python llama2_classification.py --train_sample_fraction ${sample_fraction[$sf]} & wait
done
