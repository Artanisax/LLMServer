import gc
import torch
from transformers import AutoTokenizer, AutoModel, pipeline


class LLM():
    def __init__(self, id: str):
        self.id = id
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"models/{id}",
            local_files_only=True,
            trust_remote_code=True,
        )
    
    def __del__(self):
        pass
    
    def chat(self, query: str, history: list[dict]=None):
        pass
    