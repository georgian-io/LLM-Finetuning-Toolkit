from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

class Predictor:
    def __init__(self, model_load_path: str, task: str = "summarization", load_in_8bit: bool = False):

        self.model = AutoPeftModelForCausalLM.from_pretrained(model_load_path,
                                            low_cpu_mem_usage=True,
                                            torch_dtype=torch.float16,
                                            load_in_4bit=True,)
        self.tokenizer = AutoTokenizer.from_pretrained(model_load_path)
        
        self.task = task

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

    def get_input_ids(self, prompt: str):
        if self.task == "summarization":
            input_ids = self.tokenizer("Summarize the following dialogue that is delimited with triple backticks. Dialogue: " + prompt + "Summary: ", return_tensors="pt", truncation=True).input_ids.cuda()
        else:
            input_ids = self.tokenizer(
                "Classify the following sentence that is delimited with triple backticks. Sentence: " + prompt.replace("\n", " ") + " Class: ",
                return_tensors="pt",
                truncation=True,
            ).input_ids.cuda()
        return input_ids

    @torch.inference_mode()
    def predict(self, prompt: str, max_target_length: int = 512, temperature: float = 0.01) -> str:
        input_ids = self.get_input_ids(prompt)
        outputs = self.model.generate(
            input_ids=input_ids,
            do_sample=True,
            top_p=0.9,
            max_new_tokens=max_target_length,
            temperature=temperature,
        )
        prediction = self.tokenizer.batch_decode(outputs.cpu().numpy(), skip_special_tokens=True)[0]

        return prediction
