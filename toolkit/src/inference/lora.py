import os
from os.path import join
from threading import Thread
import csv

from transformers import TextIteratorStreamer
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.text import Text
from datasets import Dataset
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch

from src.pydantic_models.config_model import Config
from src.utils.save_utils import DirectoryHelper
from src.inference.inference import Inference
from src.ui.rich_ui import RichUI


# TODO: Add type hints please!
class LoRAInference(Inference):
    def __init__(
        self,
        test_dataset: Dataset,
        label_column_name: str,
        config: Config,
        dir_helper: DirectoryHelper,
    ):
        self.test_dataset = test_dataset
        self.label_column = label_column_name
        self.config = config

        self.save_dir = dir_helper.save_paths.results
        self.save_path = join(self.save_dir, "results.csv")
        self.device_map = self.config.model.device_map
        self._weights_path = dir_helper.save_paths.weights

        self.model, self.tokenizer = self._get_merged_model(
            dir_helper.save_paths.weights
        )

    def _get_merged_model(self, weights_path: str):
        # purge VRAM
        torch.cuda.empty_cache()

        # Load from path
        dtype = (
            torch.float16
            if self.config.training.training_args.fp16
            else (
                torch.bfloat16
                if self.config.training.training_args.bf16
                else torch.float32
            )
        )

        self.model = AutoPeftModelForCausalLM.from_pretrained(
            weights_path,
            torch_dtype=dtype,
            device_map=self.device_map,
        )

        """TODO: figure out multi-gpu
        if self.config.accelerate:
            self.model = self.accelerator.prepare(self.model)
        """

        model = self.model.merge_and_unload()

        tokenizer = AutoTokenizer.from_pretrained(
            self._weights_path, device_map=self.device_map
        )

        return model, tokenizer

    def infer_all(self):
        results = []
        prompts = self.test_dataset["formatted_prompt"]
        labels = self.test_dataset[self.label_column]

        # inference loop
        for idx, (prompt, label) in enumerate(zip(prompts, labels)):
            RichUI.inference_ground_truth_display(
                f"Generating on test set: {idx+1}/{len(prompts)}", prompt, label
            )

            try:
                result = self.infer_one(prompt)
            except:
                continue
            results.append((prompt, label, result))

        # TODO: seperate this into another method
        header = ["Prompt", "Ground Truth", "Predicted"]
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in results:
                writer.writerow(row)

    def infer_one(self, prompt: str) -> str:
        input_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True
        ).input_ids.cuda()

        # stream processor
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            decode_kwargs={"skip_special_tokens": True},
            timeout=60,  # 60 sec timeout for generation; to handle OOM errors
        )

        generation_kwargs = dict(
            input_ids=input_ids, streamer=streamer, **self.config.inference.model_dump()
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        result = Text()
        with RichUI.inference_stream_display(result) as live:
            for new_text in streamer:
                result.append(new_text)
                live.update(result)

        return str(result)
