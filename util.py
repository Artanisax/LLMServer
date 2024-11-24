import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel
from datasets import load_dataset
from trl import SFTTrainer

import LLM

def create_model(id: str, base: str, lora: str) -> LLM.LLM:
    supported = {
        "ChatGLM3": LLM.ChatGLM3,
        "Gemma2": LLM.Gemma2,
        "Gemma2_IT": LLM.Gemma2_IT,
    }

def lora_finetune(id:str, base: str, dataset: str, steps: int=64) -> str:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    
    data = load_dataset(f"datasets/{dataset}")
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
    
    tokenizer = AutoTokenizer.from_pretrained(f"models/{base}")
    model = AutoModelForCausalLM.from_pretrained(
        f"models/{base}",
        quantization_config=bnb_config,
        device_map={"":0},
    )
    def formatting_func(example):
        text = f"Quote: {example['quote'][0]}\nAuthor: {example['author'][0]}\nTag: {example['tag'][0]}\n\n"
        return [text]

    trainer = SFTTrainer(
        model=model,
        train_dataset=data["train"],
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=min(10, steps // 8),
            max_steps=steps,
            learning_rate=1e-4,
            logging_steps=1,
            output_dir=f"lora/{id}",
            optim="paged_adamw_8bit"
        ),
        peft_config=lora_config,
        formatting_func=formatting_func,
    )
    trainer.train()
    return f"lora/{id}/checkpoint-{steps}"
