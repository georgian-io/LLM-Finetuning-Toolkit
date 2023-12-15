import os
import csv
from threading import Thread
import pickle

import yaml
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.text import Text

import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from transformers import TrainingArguments, TextIteratorStreamer, AutoTokenizer, ProgressCallback
from transformers.utils import logging
from trl import SFTTrainer

from src.data.dataset_generator import DatasetGenerator
from src.model.model_loader import ModelLoader
from src.utils.rich_print import inject_example_to_rich_layout

logging.set_verbosity_error()

# TODO: FUTURE BEN - Please refactor this!!!!!!!
if __name__ == "__main__":
    with open("./config.yml", "r") as file:
        config = yaml.safe_load(file)

    output_dir = os.path.join(
        config["training"].pop("output_dir", "./"),
        f"model-{config['model']['model_ckpt'].split('/')[-1]}_epochs-{config['training']['num_train_epochs']}_rank-{config['lora']['r']}_neftuneNoise-{config['sft'].get('neftune_noise_alpha', 'None')}",
    )
 
    console = Console()

    # Loading Data -------------------------------
    console.rule("[bold green]Loading Data")


    dataset_generator = DatasetGenerator(**config["data"])
    train_columns = dataset_generator.train_columns
    test_column = dataset_generator.test_column
    if not os.path.exists(f"{output_dir}/data"):
        with console.status("Injecting Columns into Prompt...", spinner="monkey"):
            train, test = dataset_generator.get_dataset()


        console.print("Saving dataset...")

        os.makedirs(f"{output_dir}/data", exist_ok=True)
        with open(f"{output_dir}/data/train.pkl", 'wb') as f:
            pickle.dump(train, f)
        with open(f"{output_dir}/data/test.pkl", 'wb') as f:
            pickle.dump(test, f)
    else:
        console.print(f"Fine-Tuned Model Found at {output_dir}... Loading Data from directory")
        with open(f"{output_dir}/data/train.pkl", "rb") as f:
            train = pickle.load(f)
        with open(f"{output_dir}/data/test.pkl", "rb") as f:
            test = pickle.load(f)

    console.print(f"Train size: {len(train)}")
    console.print(f"Test size: {len(test)}")
    # Print example
    layout = Layout()
    layout.split_row(
        Layout(Panel("Train Sample"), name="train"),
        Layout(
            Panel("Inference Sample"),
            name="inference",
        ),
    )
    inject_example_to_rich_layout(layout["train"], "Train Example", train[0])
    inject_example_to_rich_layout(layout["inference"], "Inference Example", test[0])

    console.print(layout)

    # Loading Model -------------------------------
    console.rule("[bold yellow]Loading Model")

    if not os.path.exists(f"{output_dir}/assets"):
        model, tokenizer = ModelLoader.get_model_and_tokenizer(**config["model"])
        console.print(f"{config['model']['model_ckpt']} Loaded :smile:")
    else:
        console.print(f"Fine-Tuned Model Found at {output_dir}... skipping training")


    # Injecting LoRA -------------------------------
    console.rule("[bold red]Injecting LoRA Adapters")

    if not os.path.exists(f"{output_dir}/assets"):
        config["lora"]["lora_alpha"] = (
            config["lora"].pop("alpha_factor", 2) * config["lora"]["r"]
        )
        model.gradient_checkpointing_enable()

        peft_config = LoraConfig(**config["lora"])

        if config["model"]["quant_4bit"]:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)

        console.print(f"LoRA Modules Injected")
    else:
        console.print(f"Fine-Tuned Model Found at {output_dir}... skipping training")

    # Training -------------------------------
    console.rule("[bold green]:smiley: Training")

    peft_model_id = f"{output_dir}/assets"

    if not os.path.exists(f"{output_dir}/assets"):
        logging_dir = f"{output_dir}/logs"

        training_args = TrainingArguments(
            output_dir=output_dir,
            logging_dir=logging_dir,
            report_to="none",
            **config["training"],
        )

        progress_callback = ProgressCallback()
        trainer = SFTTrainer(
            model=model,
            train_dataset=train,
            peft_config=peft_config,
            tokenizer=tokenizer,
            packing=True,
            args=training_args,
            dataset_text_field="formatted_prompt",
            callbacks=[progress_callback],
            **config["sft"],
        )

        with console.status("Training...", spinner="runner"):
            trainer_stats = trainer.train()
        console.print(f"Training Complete")

        trainer.model.save_pretrained(peft_model_id)
        tokenizer.save_pretrained(peft_model_id)
        console.print(f"Run saved at {output_dir}")

    else:
        console.print(f"Fine-Tuned Model Found at {output_dir}... skipping training")

    # Inference -------------------------------
    console.rule("[bold pink]:face_with_monocle: Testing")

    console.print("Merging Adapter Weights...")
    torch.cuda.empty_cache()
    model = AutoPeftModelForCausalLM.from_pretrained(
        peft_model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        )
    model = model.merge_and_unload()
    console.print("Done Merging")

    tokenizer = AutoTokenizer.from_pretrained(peft_model_id, device_map="auto")

    results = []
    oom_examples = []
    prompts, labels = test["formatted_prompt"], test[test_column]


    for idx, (prompt, label) in enumerate(zip(prompts, labels)):
        inf_table = Table(title=f"Inference Results {idx+1}/{len(prompts)}", show_lines=True)
        inf_table.add_column("prompt")
        inf_table.add_column("ground truth")
        inf_table.add_row(prompt, label)
        console.print(inf_table)

        input_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True
        ).input_ids.cuda(
        )

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, decode_kwargs={"skip_special_tokens":True})
        generation_kwargs = dict( 
            input_ids=input_ids,
            streamer=streamer,
            **config["inference"]
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        console.print("[bold red]Prediction >")
        result = Text()
        with Live(result, refresh_per_second=4, vertical_overflow="visible") as live:
            for new_text in streamer:
                result.append(new_text)
                live.update(result)

        results.append((prompt, label, str(result)))
    
    console.print("saving inference results...")

    header = ["Prompt", "Ground Truth", "Predicted"]
    with open(f"{output_dir}/output.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in results:
            writer.writerow(row)