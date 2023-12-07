import os
import csv

import yaml
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Console
from rich.table import Table
from rich.live import Live

import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer

from src.data.dataset_generator import DatasetGenerator
from src.model.model_loader import ModelLoader
from src.utils.rich_print import inject_example_to_rich_layout


if __name__ == "__main__":
    console = Console()

    # Loading Data -------------------------------
    console.rule("[bold green]Loading Data")

    with open("./config.yml", "r") as file:
        config = yaml.safe_load(file)

    dataset_generator = DatasetGenerator(**config["data"])
    train_columns = dataset_generator.train_columns
    test_column = dataset_generator.test_column

    with console.status("Injecting Columns into Prompt...", spinner="monkey"):
        train, test = dataset_generator.get_dataset()

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
    model, tokenizer = ModelLoader.get_model_and_tokenizer(**config["model"])
    console.print(f"{config['model']['model_ckpt']} Loaded :smile:")

    # Injecting LoRA -------------------------------
    console.rule("[bold red]Injecting LoRA Adapters")
    config["lora"]["lora_alpha"] = (
        config["lora"].pop("alpha_factor", 2) * config["lora"]["r"]
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    peft_config = LoraConfig(**config["lora"])

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    console.print(f"LoRA Modules Injected")

    # Training -------------------------------
    console.rule("[bold green]:smiley:Training")
    output_dir = os.path.join(
        config["training"].pop("output_dir", "./"),
        f"model-{config['model']['model_ckpt'].split('/')[-1]}_epochs-{config['training']['num_train_epochs']}_rank-{config['lora']['r']}_neftuneNoise-{None}",
    )
    logging_dir = f"{output_dir}/logs"

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        report_to="none",
        **config["training"],
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train,
        peft_config=peft_config,
        tokenizer=tokenizer,
        packing=True,
        args=training_args,
        dataset_text_field="formatted_prompt",
        **config["sft"],
    )

    trainer_stats = trainer.train()
    console.print(f"Training Complete")

    peft_model_id = f"{output_dir}/assets"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
    console.print(f"Run saved at {output_dir}")

    # Inference -------------------------------
    console.rule("[bold pink]:face_with_monocle: Testing")
    model.eval()

    results = []
    oom_examples = []
    prompts, labels = test["formatted_prompt"], test[test_column]


    for prompt, label in zip(prompts, labels):
        inf_table = Table(title="Inference Results", show_lines=True)
        inf_table.add_column("prompt")
        inf_table.add_column("ground truth")
        inf_table.add_column("predicted")

        input_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True
        ).input_ids.cuda()

        with torch.inference_mode():
            try:
                outputs = model.generate(
                    input_ids=input_ids, **config["inference"]
                )
                result = tokenizer.batch_decode(
                    outputs.detach().cpu().numpy(), skip_special_tokens=True
                )[0]
                result = result[len(prompt) :]
            except:
                result = "[OOM ERROR]"
                oom_examples.append(input_ids.shape[-1])


        results.append((prompt, label, result))
        inf_table.add_row(prompt, label, result)
        console.print(inf_table)
    
    console.print("saving inference results...")

    header = ["Prompt", "Ground Truth", "Predicted"]
    with open(f"{output_dir}/output.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in results:
            writer.writerow(row)