from threading import Thread
import csv

from transformers import TextIteratorStreamer
from rich.table import Table
from rich.live import Live
from rich.text import Text


# TODO: Add type hints please!
class InferenceRunner:
    def __init__(
        self,
        model,
        tokenizer,
        test_dataset,
        label_column_name,
        config,
        console,
        save_path,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.test = test_dataset
        self.label_column = label_column_name
        self.config = config
        self.save_path = save_path
        self._console = console

    def _init_rich_table(self, title: str, prompt: str, label: str) -> Table:
        table = Table(title=title, show_lines=True)
        table.add_column("prompt")
        table.add_column("ground truth")
        table.add_row(prompt, label)

        return table

    def run_inference(self):
        results = []
        prompts = self.test["formatted_prompt"]
        labels = self.test[self.label_column_name]

        # inference loop
        for idx, (prompt, label) in enumerate(zip(prompts, labels)):
            table = self._init_rich_table(
                f"Generating on test set: {idx+1}/{len(prompts)}"
            )
            self._console.print(table)

            input_ids = self.tokenizer(
                prompt, return_tensors="pt", truncation=True
            ).input_ids.cuda()

            # stream processor
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                decode_kwargs={"skip_special_tokens": True},
            )

            generation_kwargs = dict(
                input_ids=input_ids, streamer=streamer, **self.config.inference
            )

            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            self._console.print("[bold red]Prediction >")
            result = Text()

            with Live(
                result, refresh_per_second=4, vertical_overflow="visible"
            ) as live:
                for new_text in streamer:
                    result.append(new_text)
                    live.update(result)

            results.append((prompt, label, str(result)))

        self._console.print("Saving inference results...")
        header = ["Prompt", "Ground Truth", "Predicted"]
        with open(self.save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in results:
                writer.writerow(row)
