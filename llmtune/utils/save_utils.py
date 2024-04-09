"""
Helper functions to help managing saving and loading of experiments:
    1. Generate save directory name
    2. Check if files are present at various experiment stages
"""

import hashlib
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import yaml
from sqids import Sqids

from llmtune.constants.files import (
    CONFIG_DIR_NAME,
    CONFIG_FILE_NAME,
    DATASET_DIR_NAME,
    NUM_MD5_DIGITS_FOR_SQIDS,
    QA_DIR_NAME,
    QA_FILE_NAME,
    RESULTS_DIR_NAME,
    RESULTS_FILE_NAME,
    WEIGHTS_DIR_NAME,
)
from llmtune.pydantic_models.config_model import Config


@dataclass
class DirectoryList:
    save_dir: Path
    config_hash: str

    @property
    def experiment(self) -> Path:
        return self.save_dir / self.config_hash

    @property
    def config(self) -> Path:
        return self.experiment / CONFIG_DIR_NAME

    @property
    def config_file(self) -> Path:
        return self.config / CONFIG_FILE_NAME

    @property
    def dataset(self) -> Path:
        return self.experiment / DATASET_DIR_NAME

    @property
    def weights(self) -> Path:
        return self.experiment / WEIGHTS_DIR_NAME

    @property
    def results(self) -> Path:
        return self.experiment / RESULTS_DIR_NAME

    @property
    def results_file(self) -> Path:
        return self.results / RESULTS_FILE_NAME

    @property
    def qa(self) -> Path:
        return self.experiment / QA_DIR_NAME

    @property
    def qa_file(self) -> Path:
        return self.qa / QA_FILE_NAME


class DirectoryHelper:
    def __init__(self, config_path: Path, config: Config):
        self.config_path: Path = config_path
        self.config: Config = config
        self.sqids: Sqids = Sqids()
        self.save_paths: DirectoryList = self._get_directory_state()

        self.save_paths.experiment.mkdir(parents=True, exist_ok=True)
        if not self.save_paths.config.exists():
            self.save_config()

    @cached_property
    def config_hash(self) -> str:
        config_str = self.config.model_dump_json()
        config_str = re.sub(r"\s", "", config_str)
        hash = hashlib.md5(config_str.encode()).digest()
        return self.sqids.encode(hash[:NUM_MD5_DIGITS_FOR_SQIDS])

    def _get_directory_state(self) -> DirectoryList:
        save_dir = (
            Path(self.config.save_dir)
            if not self.config.ablation.use_ablate
            else Path(self.config.save_dir) / self.config.ablation.study_name
        )
        return DirectoryList(save_dir, self.config_hash)

    def save_config(self) -> None:
        self.save_paths.config.mkdir(parents=True, exist_ok=True)
        model_dict = self.config.model_dump()

        with (self.save_paths.config / "config.yml").open("w") as file:
            yaml.dump(model_dict, file)
