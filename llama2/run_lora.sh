epochs=(2 5 10 20 30 50)
lora_r=(2 4 8 16)
dropout=(0.1 0.2 0.5)

for (( epoch=0; epoch<6; epoch=epoch+1 )) do
	for ((r=0; r<4; r=r+1 )) do
		for (( d=0; d<3; d=d+1 )) do
			python llama2_summarization.py --lora_r ${lora_r[$r]} --epochs ${epochs[$epoch]} --dropout ${dropout[$d]} & wait
		done
	done
done
