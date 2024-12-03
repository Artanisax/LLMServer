import gc
import os
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from .llm import LLM


class Gemma_2_IT(LLM):
    def __init__(self, id: str = "google/gemma-2-2b-it", lora: str = None):
        super().__init__(id)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            os.path.join("models", id),
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            trust_remote_code=True,
            device_map="cuda",
        ).eval()

    def __del__(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def chat(self, query: str, history: list[dict] = None):
        input_ids = self.tokenizer(query, return_tensors="pt").to("cuda")
        num_tokens = input_ids["input_ids"].size(-1)
        outputs = self.model.generate(
            **input_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        num_tokens += outputs.size(-1)
        
        return {
            "response": response,
            "history": None,
            "num_tokens": num_tokens,
        }
