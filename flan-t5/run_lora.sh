epochs=(2 5 10)
lora_r=(2 4 8 16)
dropout=(0.1 0.2)

for (( epoch=0; epoch<3; epoch=epoch+1 )) do
	for ((r=0; r<4; r=r+1 )) do
		for (( d=0; d<2; d=d+1 )) do
			python flan_classification.py --lora_r ${lora_r[$r]} --epochs ${epochs[$epoch]} --dropout ${dropout[$d]} & wait
		done
	done
done
