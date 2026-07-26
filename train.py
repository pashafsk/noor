from unsloth import FastLanguageModel

import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

MAX_SEQ_LENGTH = 2048

# -----------------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------------

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/workspace/models/qwen2.5-7b",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

dataset = load_dataset(
    "json",
    data_files="/workspace/nasdaq100_traps.jsonl",
    split="train",
)

# -----------------------------------------------------------------------------
# REQUIRED by Unsloth
# -----------------------------------------------------------------------------

def formatting_func(example):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

# -----------------------------------------------------------------------------
# Training args
# -----------------------------------------------------------------------------

training_args = SFTConfig(
    output_dir="outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=60,
    learning_rate=2e-4,
    logging_steps=1,
    save_steps=60,
    save_strategy="steps",
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    report_to="none",
)

# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    formatting_func=formatting_func,
)

# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------

trainer.train()

# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------

trainer.save_model("outputs/final")
tokenizer.save_pretrained("outputs/final")
