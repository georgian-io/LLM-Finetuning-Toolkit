sample_fraction=(0.025 0.05 0.1 0.25 0.5)

for (( sf=0; sf<5; sf=sf+1 )) do
	python falcon_classification.py --train_sample_fraction ${sample_fraction[$sf]} & wait
done
