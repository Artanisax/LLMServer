import os
from transformers import AutoTokenizer
from torch import Tensor

class LLM():
    ROOT = "/home/artanisax/Projects/OOAD/LLMServer"
    
    def __init__(self, id: str):
        self.id = id
        self.max_new_tokens = 256
        self.temperature = 0.7
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join("models", id),
            local_files_only=True,
            trust_remote_code=True,
        )
    
    def __del__(self):
        del self.tokenizer
    
    def config(self, attr: dict):
        for key, value in attr.items():
            self.__setattr__(key, value)
        
    def chat(self, query: str, history: list[dict]=None):
        pass
    
