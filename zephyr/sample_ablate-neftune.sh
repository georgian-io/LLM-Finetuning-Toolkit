sample_fraction=(0.025 0.05)

for (( sf=0; sf<3; sf=sf+1 )) do
	python zephyr_classification.py --train_sample_fraction ${sample_fraction[$sf]} --neftune 5.0 & wait
done
