"""
Helper functions to help managing saving and loading of experiments:
    1. Generate save directory name
    2. Check if files are present at various experiment stages
"""

import shutil
import os
from os.path import exists
import yaml

import re
import hashlib
from functools import cached_property
from dataclasses import dataclass

from sqids import Sqids

from src.pydantic_models.config_model import Config

NUM_MD5_DIGITS_FOR_SQIDS = 5  # TODO: maybe move consts to a dedicated folder


@dataclass
class DirectoryList:
    save_dir: str
    config_hash: str

    @property
    def experiment(self) -> str:
        return os.path.join(self.save_dir, self.config_hash)

    @property
    def config(self) -> str:
        return os.path.join(self.experiment, "config")

    @property
    def dataset(self) -> str:
        return os.path.join(self.experiment, "dataset")

    @property
    def weights(self) -> str:
        return os.path.join(self.experiment, "weights")

    @property
    def results(self) -> str:
        return os.path.join(self.experiment, "results")

    @property
    def qa(self) -> str:
        return os.path.join(self.experiment, "qa")


class DirectoryHelper:
    def __init__(self, config_path: str, config: Config):
        self.config_path: str = config_path
        self.config: Config = config
        self.sqids: Sqids = Sqids()
        self.save_paths: DirectoryList = self._get_directory_state()

        os.makedirs(self.save_paths.experiment, exist_ok=True)
        if not exists(self.save_paths.config):
            self.save_config()

    @cached_property
    def config_hash(self) -> str:
        config_str = self.config.model_dump_json()
        config_str = re.sub(r"\s", "", config_str)
        hash = hashlib.md5(config_str.encode()).digest()
        return self.sqids.encode(hash[:NUM_MD5_DIGITS_FOR_SQIDS])

    def _get_directory_state(self) -> DirectoryList:
        save_dir = (
            self.config.save_dir
            if not self.config.ablation.use_ablate
            else os.path.join(self.config.save_dir, self.config.ablation.study_name)
        )
        return DirectoryList(save_dir, self.config_hash)

    def save_config(self) -> None:
        os.makedirs(self.save_paths.config, exist_ok=True)
        model_dict = self.config.model_dump()

        with open(os.path.join(self.save_paths.config, "config.yml"), "w") as file:
            yaml.dump(model_dict, file)
