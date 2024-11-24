import gc
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from .llm import LLM

class Gemma2_IT(LLM):
    def __init__(self, id: str, lora: str=None):
        super().__init__(id)
        self.max_new_tokens = 256
        self.model = AutoModelForCausalLM.from_pretrained(
            f"models/{id}",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            trust_remote_code=True,
            device_map="auto",
        ).eval()
    
    def __del__(self):
        del self.tokenizer, self.model
        gc.collect()
        torch.cuda.empty_cache()
    
    def chat(self, query: str, history: list[dict]=None):
        input_ids = self.tokenizer(query, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        return {
            "response": self.tokenizer.decode(outputs[0]),
            "history": None,
        }
    