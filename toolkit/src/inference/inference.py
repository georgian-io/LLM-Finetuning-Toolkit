from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Dict


class Inference(ABC):
    @abstractmethod
    def infer_one(self, prompt: str):
        pass

    @abstractmethod
    def infer_all(self):
        pass
