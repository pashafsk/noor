from datasets import load_dataset
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

max_seq_length = 4096
dtype = None
load_in_4bit = True

# 1. Load your base model from your models directory
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/workspace/models/qwen2.5-7b",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# 2. Setup QLoRA adapters
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
    random_state=3407,
)

# 3. Load your dataset from /workspace
dataset = load_dataset(
    "json", data_files="/workspace/nasdaq100_traps.jsonl", split="train"
)

# 4. Configure Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
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

# 5. Train
trainer_stats = trainer.train()

# 6. Save locally
model.save_pretrained_merged(
    "qwen2.5-7b-trader-adapter", tokenizer, save_method="merged_16bit"
)
print("Training complete and model merged successfully!")
