"""
Helper functions to help managing saving and loading of experiments:
    1. Generate save directory name
    2. Check if files are present at various experiment stages
"""
import shutil
import os
from os.path import exists

import re
import hashlib
from functools import cached_property
from dataclasses import dataclass

from sqids import Sqids

from src.pydantic_models.config_model import Config

NUM_MD5_DIGITS_FOR_SQIDS = 5


@dataclass
class DirectoryList:
    save_dir: str
    config_hash: str

    @property
    def experiment_path(self) -> str:
        return os.path.join(self.save_dir, self.config_hash)

    @property
    def config_path(self) -> str:
        return os.path.join(self.experiment_path, "/config.yml")

    @property
    def dataset_path(self) -> str:
        return os.path.join(self.experiment_path, "/dataset")

    @property
    def weights_path(self) -> str:
        return os.path.join(self.experiment_path, "/weights")

    @property
    def results_path(self) -> str:
        return os.path.join(self.experiment_path, "/results")


class DirectoryHelper:
    def __init__(self, config_path: str, config: Config):
        self.config_path: str = config_path
        self.config: Config = config
        self.sqids: Sqids = Sqids()
        self.save_paths: DirectoryList = self._get_directory_state()

        os.makedirs(self.save_paths.experiment_path, exist_ok=True)
        if not exists(self.save_paths.config_path):
            self.save_config()

    @cached_property
    def config_hash(self) -> str:
        with open(self.config_path) as f:
            config_str = f.read()
        config_str = re.sub(r"\s", "", config_str)
        hash = hashlib.md5(config_str.encode()).digest()
        return self.sqids.encode(hash[:NUM_MD5_DIGITS_FOR_SQIDS])

    def _get_directory_state(self) -> DirectoryList:
        return DirectoryList(self.config.save_dir, self.config_hash)

    def save_config(self) -> None:
        os.makedirs(self.save_paths.config_path, exist_ok=True)
        shutil.copy(self.config_path, self.save_paths.config_path, "config.yml")
