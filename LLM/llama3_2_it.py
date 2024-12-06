import gc
import os
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, pipeline

from .llm import LLM


class Llama_3_2_Instruct(LLM):
    def __init__(self, id: str = "Llama-3.2-1B-Instruct", lora: str = None):
        super().__init__(id)
        self.system = "You are a helpful assistant."

        self.model = AutoModelForCausalLM.from_pretrained(
            os.path.join("models", id),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=True,
            device_map="cuda",
        ).eval()
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="cuda",
        )

    def __del__(self):
        del self.model, self.pipe
        gc.collect()
        torch.cuda.empty_cache()

    def chat(self, query: str, history: list[dict] = None):
        if history is None:
            history = []
        message = (
            [{"role": "system", "content": self.system}]
            + history
            + [{"role": "user", "content": query}]
        )
        outputs = self.pipe(
            message,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        response = outputs[0]["generated_text"][-1]["content"]
        num_tokens = (
            len(self.tokenizer(query)["input_ids"])
            + len(self.tokenizer(response)["input_ids"])
        )

        return {
            "response": response,
            "history": None,
            "num_tokens": num_tokens,
        }
