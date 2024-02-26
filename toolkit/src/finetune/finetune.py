from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Dict


class Finetune(ABC):
    @abstractmethod
    def finetune(self):
        pass

    @abstractmethod
    def save_model(self):
        pass