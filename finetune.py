import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from datasets import load_dataset
from trl import SFTTrainer


def main(args):
    id = args.id
    dataset = args.dataset
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(f"models/{id}")
    model = AutoModelForCausalLM.from_pretrained(
        f"models/{id}",
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    data = load_dataset(f"datasets/{dataset}")
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
    lora_config = LoraConfig(
        r=8,
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    def formatting_func(example):
        text = f"Quote: {example['quote'][0]}\n\nAuthor: {example['author'][0]}\n\nTag: {example['tag'][0]}"
        return [text]

    trainer = SFTTrainer(
        model=model,
        train_dataset=data["train"],
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=args.steps,
            learning_rate=1e-4,
            logging_steps=1,
            output_dir=f"lora/{args.name}",
            optim="paged_adamw_8bit",
        ),
        peft_config=lora_config,
        formatting_func=formatting_func,
    )
    trainer.train()
    lora_name = f"{args.name}/checkpoint-{args.steps}"
    print(lora_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True)
    parser.add_argument("--id", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()
    main(args)
