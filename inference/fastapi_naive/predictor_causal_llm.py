from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

class Predictor:
    def __init__(self, model_load_path: str, task: str = "summarization", load_in_8bit: bool = False):
        config = PeftConfig.from_pretrained(model_load_path)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, 
                                                          load_in_8bit=load_in_8bit,
                                                          quantization_config=bnb_config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.task = task
        self.model = PeftModel.from_pretrained(self.model, model_load_path)

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
