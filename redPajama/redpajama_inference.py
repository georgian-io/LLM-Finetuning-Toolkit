import torch

from falcon_seq2seq import get_data, preprocess_function

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
)

from peft import (
    PeftConfig,
    PeftModel,
)

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score
)

experiment = "lora-2-4-0.1"
peft_model_id = f"experiments/{experiment}/assets"

config = PeftConfig.from_pretrained(peft_model_id)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, 
    quantization_config=bnb_config, 
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, peft_model_id)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

dataset_train, dataset_val = get_data()
tokenized_dataset_val = dataset_val.map(
    preprocess_function, batched=True, remove_columns=["input_text", "output_text"],
    fn_kwargs={"tokenizer": tokenizer},
)
instructions = tokenized_dataset_val["instructions"]
labels = dataset_val["output_text"]

results = []
for instruct, label in zip(instructions, labels):
    example = instruct[:-len(label)] # remove the answer from the example
    input_ids = tokenizer(example, return_tensors="pt", truncation=True).input_ids.cuda()
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=200, do_sample=True, top_p=0.95, temperature=1e-3
        )
        result = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
        result = result.split("The answer is:")[-1].lstrip()
        results.append(result)

import ipdb; ipdb.set_trace()

metrics = {
    "micro_f1": f1_score(labels, results, average="micro"),
    "macro_f1": f1_score(labels, results, average="macro"),
    "precision": precision_score(labels, results, average="micro"),
    "recall": recall_score(labels, results, average="micro"),
    "accuracy": accuracy_score(labels, results),
}

print(metrics)


metrics["results"] = results
metrics["labels"] = labels

import os
import pickle
with open(os.path.join(peft_model_id, "metrics.pkl"), "wb") as handle:
    pickle.dump(metrics, handle)


