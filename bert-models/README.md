# Contents:

We know, We know!

Not another BERT finetuning script! 

We use the script ```train.py``` to benchmark BERT-like models on the News Group classification dataset so we thought of adding it to this repository for reproducibility purposes :) 

The shell script ```ablate.sh``` finetunes a BERT-like model on different number of training samples, controlled by variable __sample_fraction__. We do this experiment to stress-test BERT-like models and evaluate their sample efficiency.  

Please feel free to skip this folder if you are already aware of how to finetune BERT-like models!

Alternatively, you can always come back to this folder if you want to quickly take out BERT models for a spin on your dataset! :)

Simply do ```python train.py --model bert-base-uncased``` or ```python train.py --model distilbert-base-uncased``` or any other transformer model that's hosted on HuggingFace.

Happy finetuning BERT! 

<img src="../images/bert.gif" width="128" height="128"/>
