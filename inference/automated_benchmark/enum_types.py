from enum import Enum

class Server(Enum):
    TGI = "tgi"
    VLLM = "vllm"
    RAY = "ray"

class Task(Enum):
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"