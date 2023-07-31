from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


class Predictor:
    def __init__(self, model_load_path: str, task: str = "summarization", load_in_8bit: bool = False):
        config = PeftConfig.from_pretrained(model_load_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, load_in_8bit=load_in_8bit)
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

        self.task = task
        self.model = PeftModel.from_pretrained(self.model, model_load_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

    def get_input_ids(self, prompt: str):
        if self.task == "summarization":
            input_ids = self.tokenizer("summarize: " + prompt, return_tensors="pt", truncation=True).input_ids.cuda()
        else:
            input_ids = self.tokenizer(
                "Classify the following sentence into a category: " + prompt.replace("\n", " ") + " The answer is: ",
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