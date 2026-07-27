from unsloth import FastLanguageModel

max_seq_length = 2048

# 1. Load your fine-tuned adapters from the outputs directory
# (If 'outputs' has subfolders, change this to the exact checkpoint path, e.g., "outputs/checkpoint-60")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs", 
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

# 2. Save and Export the Merged Model
# This exports a full 16-bit model with the LoRA adapter merged into the base weights
print("Merging to 16-bit and saving...")
model.save_pretrained_merged("qwen2.5-7b-finetuned-16bit", tokenizer, save_method="merged_16bit")

# OPTIONAL: If you plan to run this via serverless deployment using Ollama or llama.cpp, 
# uncomment the line below to export directly to a GGUF format instead.
# model.save_pretrained_gguf("qwen2.5-7b-finetuned-gguf", tokenizer, quantization_method="q4_k_m")
