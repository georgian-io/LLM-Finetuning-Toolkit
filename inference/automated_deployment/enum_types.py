from enum import Enum

class Server(Enum):
    TGI = "tgi"
    VLLM = "vllm"
    RAY = "ray"
    TRITON_VLLM = "triton_vllm"

class Task(Enum):
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"