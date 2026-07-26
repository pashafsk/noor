from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/workspace/models/qwen2.5-7b",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# Load the dataset from the specified path in /workspace
dataset = load_dataset("json", data_files={"train": "/workspace/nasdaq100_traps.jsonl"}, split="train")

# Define formatting function for chat messages structure
def formatting_prompts_func(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)}

# Map formatting across the dataset
dataset = dataset.map(formatting_prompts_func, batched=False)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        output_dir="outputs",
    ),
)

trainer_stats = trainer.train()
