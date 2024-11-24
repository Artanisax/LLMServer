import gc
import torch
from llm import LLM
from transformers import AutoTokenizer, AutoModel, pipeline


class Chatglm3(LLM):
    def __init__(self, id: str):
        super.__init__(id)
        self.model = AutoModel.from_pretrained(
            f"models/{id}",
            load_in_4bit=True,
            local_files_only=True,
            trust_remote_code=True,
        ).cuda().eval()
    
    def __del__(self):
        del self.model, self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        self.model, self.tokenizer = None, None
    
    def chat(self, query: str, history: list[dict]=None):
        response, history = self.model.chat(self.tokenizer, query, history)
        return {
            "response": response,
            "history": history,
        }
    