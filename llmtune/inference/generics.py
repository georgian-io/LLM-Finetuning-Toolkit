from abc import ABC, abstractmethod


class Inference(ABC):
    @abstractmethod
    def infer_one(self, prompt: str):
        pass

    @abstractmethod
    def infer_all(self):
        pass
