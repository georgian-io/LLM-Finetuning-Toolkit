from datasets import Dataset

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text

from src.ui.ui import UI
from src.utils.rich_print_utils import inject_example_to_rich_layout

console = Console()


class StatusContext:
    def __init__(self, console, message, spinner):
        self.console = console
        self.message = message
        self.spinner = spinner

    def __enter__(self):
        self.task = self.console.status(self.message, spinner=self.spinner)
        self.task.__enter__()  # Manually enter the console status context
        return self  # This allows you to use variables from this context if needed

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.task.__exit__(
            exc_type, exc_val, exc_tb
        )  # Cleanly exit the console status context


class LiveContext:
    def __init__(self, text: Text, refresh_per_second=4, vertical_overflow="visible"):
        self.console = console
        self.text = text
        self.refresh_per_second = refresh_per_second
        self.vertical_overflow = vertical_overflow

    def __enter__(self):
        self.task = Live(
            self.text,
            refresh_per_second=self.refresh_per_second,
            vertical_overflow=self.vertical_overflow,
        )
        self.task.__enter__()  # Manually enter the console status context
        return self  # This allows you to use variables from this context if needed

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.task.__exit__(
            exc_type, exc_val, exc_tb
        )  # Cleanly exit the console status context

    def update(self, new_text: Text):
        self.task.update(new_text)


class RichUI(UI):
    """
    DATASET
    """

    # Lifecycle functions
    @staticmethod
    def before_dataset_creation():
        console.rule("[bold green]Loading Data")

    @staticmethod
    def during_dataset_creation(message: str, spinner: str):
        return StatusContext(console, message, spinner)

    @staticmethod
    def after_dataset_creation(save_dir: str, train: Dataset, test: Dataset):
        console.print(f"Dataset Saved at {save_dir}")
        console.print(f"Post-Split data size:")
        console.print(f"Train: {len(train)}")
        console.print(f"Test: {len(test)}")

    @staticmethod
    def dataset_found(save_dir: str):
        console.print(f"Loading formatted dataset from directory {save_dir}")

    # Display functions
    @staticmethod
    def dataset_display_one_example(train_row: dict, test_row: dict):
        layout = Layout()
        layout.split_row(
            Layout(Panel("Train Sample"), name="train"),
            Layout(
                Panel("Inference Sample"),
                name="inference",
            ),
        )

        inject_example_to_rich_layout(layout["train"], "Train Example", train_row)
        inject_example_to_rich_layout(
            layout["inference"], "Inference Example", test_row
        )

        console.print(layout)

    """
    FINETUNING
    """

    # Lifecycle functions
    @staticmethod
    def before_finetune():
        console.rule("[bold yellow]:smiley: Finetuning")

    @staticmethod
    def on_basemodel_load(checkpoint: str):
        console.print(f"Loading {checkpoint}...")

    @staticmethod
    def after_basemodel_load(checkpoint: str):
        console.print(f"{checkpoint} Loaded :smile:")

    @staticmethod
    def during_finetune():
        return StatusContext(console, "Finetuning Model...", "runner")

    @staticmethod
    def after_finetune():
        console.print(f"Finetuning complete!")

    @staticmethod
    def finetune_found(weights_path: str):
        console.print(f"Fine-Tuned Model Found at {weights_path}... skipping training")

    """
    INFERENCE 
    """

    # Lifecycle functions
    @staticmethod
    def before_inference():
        console.rule("[bold pink]:face_with_monocle: Testing")

    @staticmethod
    def during_finetune():
        pass

    @staticmethod
    def after_inference(results_path: str):
        console.print(f"Inference Results Saved at {results_path}")

    @staticmethod
    def results_found(results_path: str):
        console.print(f"Inference Results Found at {results_path}")

    # Display functions
    @staticmethod
    def inference_ground_truth_display(title: str, prompt: str, label: str):
        table = Table(title=title, show_lines=True)
        table.add_column("prompt")
        table.add_column("ground truth")
        table.add_row(prompt, label)
        console.print(table)

    @staticmethod
    def inference_stream_display(text: Text):
        console.print("[bold red]Prediction >")
        return LiveContext(text)

    """
    QA 
    """

    # Lifecycle functions
    @staticmethod
    def before_qa():
        pass

    @staticmethod
    def during_qa():
        pass

    @staticmethod
    def after_qa():
        pass

    @staticmethod
    def qa_found():
        pass
    
    @staticmethod
    def qa_display_table(self, result_dictionary, mean_values, median_values, stdev_values):
        
        # Create a table
        table = Table(show_header=True, header_style="bold", title="Test Results")

        # Add columns to the table
        table.add_column("Metric", style="cyan")
        table.add_column("Mean", style="magenta")
        table.add_column("Median", style="green")
        table.add_column("Standard Deviation", style="yellow")

        # Add data rows to the table
        for key in result_dictionary:
            table.add_row(
                key,
                f"{mean_values[key]:.4f}",
                f"{median_values[key]:.4f}",
                f"{stdev_values[key]:.4f}"
            )

        # Print the table
        console.print(table)
