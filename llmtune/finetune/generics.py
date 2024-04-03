from abc import ABC, abstractmethod


class Finetune(ABC):
    @abstractmethod
    def finetune(self):
        pass

    @abstractmethod
    def save_model(self):
        pass
