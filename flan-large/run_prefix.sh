epochs=(5 10 15 20)
prefix_tokens=(10 25 50 100)
prefix_projection=(0 1)

for (( epoch=0; epoch<4; epoch=epoch+1 )) do
	for ((pt=0; pt<4; pt=pt+1 )) do
		for (( proj=0; proj<2; proj=proj+1 )) do
			python flan_seq2seq.py --prefix_tokens ${prefix_tokens[$pt]} --epochs ${epochs[$epoch]} --prefix_projection ${prefix_projection[$proj]} & wait
		done
	done
done
