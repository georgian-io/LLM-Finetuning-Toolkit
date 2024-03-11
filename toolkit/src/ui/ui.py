from abc import ABC, abstractstaticmethod

from datasets import Dataset
from rich.text import Text


class UI(ABC):
    """
    DATASET
    """

    # Lifecycle functions
    @abstractstaticmethod
    def before_dataset_creation():
        pass

    @abstractstaticmethod
    def during_dataset_creation(message: str, spinner: str):
        pass

    @abstractstaticmethod
    def after_dataset_creation(save_dir: str, train: Dataset, test: Dataset):
        pass

    @abstractstaticmethod
    def dataset_found(save_dir: str):
        pass

    # Display functions
    @abstractstaticmethod
    def dataset_display_one_example(train_row: dict, test_row: dict):
        pass

    """
    FINETUNING
    """

    # Lifecycle functions
    @abstractstaticmethod
    def before_finetune():
        pass

    @abstractstaticmethod
    def on_basemodel_load(checkpoint: str):
        pass

    @abstractstaticmethod
    def after_basemodel_load(checkpoint: str):
        pass

    @abstractstaticmethod
    def during_finetune():
        pass

    @abstractstaticmethod
    def after_finetune():
        pass

    @abstractstaticmethod
    def finetune_found(weights_path: str):
        pass

    """
    INFERENCE 
    """

    # Lifecycle functions
    @abstractstaticmethod
    def before_inference():
        pass

    @abstractstaticmethod
    def during_inference():
        pass

    @abstractstaticmethod
    def after_inference(results_path: str):
        pass

    @abstractstaticmethod
    def results_found(results_path: str):
        pass

    # Display functions
    @abstractstaticmethod
    def inference_ground_truth_display(title: str, prompt: str, label: str):
        pass

    @abstractstaticmethod
    def inference_stream_display(text: Text):
        pass

    """
    QA 
    """

    # Lifecycle functions
    @abstractstaticmethod
    def before_qa(cls):
        pass

    @abstractstaticmethod
    def during_qa(cls):
        pass

    @abstractstaticmethod
    def after_qa(cls):
        pass

    @abstractstaticmethod
    def qa_found(cls):
        pass
    
    @abstractstaticmethod
    def qa_display_table(cls):
        pass