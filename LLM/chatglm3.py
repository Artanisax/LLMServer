import gc
import torch
from transformers import AutoModel

from .llm import LLM

class ChatGLM3(LLM):
    def __init__(self, id: str, lora: str=None):
        super().__init__(id)
        self.max_new_tokens = 256
        self.num_beams = 1
        self.temperature = 0.8
        self.model = AutoModel.from_pretrained(
            f"models/{id}",
            load_in_4bit=True,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
    
    def __del__(self):
        del self.tokenizer, self.model
        gc.collect()
        torch.cuda.empty_cache()
    
    def chat(self, query: str, history: list[dict]=None):
        response, history = self.model.chat(
            self.tokenizer,
            query,
            history,
            max_length=self.max_new_tokens,
            num_beams=self.num_beams,
            temperature=self.temperature,
        )
        return {
            "response": response,
            "history": history,
        }
    