import gc
import torch
from transformers import AutoTokenizer, AutoModel, pipeline


class LLM():
    def __init__(self, id: str):
        self.id = id
        self.tokenizer, self.model, self.pipeline = None, None, None
        supported = [
            "THUDM/chatglm3-6b",
            "databricks/dolly-v2-3b",
        ]
        if id in supported:
            self.tokenizer = AutoTokenizer.from_pretrained(
                    f"../models/{id}",
                    local_files_only=True,
                    trust_remote_code=True,
                )
            if id == "../THUDM/chatglm3-6b":
                self.model = AutoModel.from_pretrained(
                    f"models/{id}",
                    load_in_4bit=True,
                    local_files_only=True,
                    trust_remote_code=True,
                ).cuda().eval()
            elif id == "databricks/dolly-v2-3b":
                self.pipeline = pipeline(
                    model="databricks/dolly-v2-3b",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="auto"
                )
        else:
            raise NotImplementedError
    
    def __del__(self):
        del self.model, self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        self.model, self.tokenizer = None, None
    
    def chat(self, query: str, history: list[dict]=None):
        pass
        response, history = self.model.chat(self.tokenizer, query, history)
        return {
            "response": response,
            "history": history,
        }
    