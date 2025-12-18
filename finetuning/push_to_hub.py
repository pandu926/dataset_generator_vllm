"""
Merge LoRA adapters with base model and push to HuggingFace
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configuration
BASE_MODEL = "google/gemma-3-1b-it"
ADAPTER_PATH = "./outputs/gemma3-1b-qlora-sft/final_model"
MERGED_PATH = "./outputs/gemma3-1b-merged"
HF_REPO = "Pandusu/gemma-3-1b-pmb-qlora-multiturn"

print("="*60)
print("MERGE LORA AND PUSH TO HUGGINGFACE")
print("="*60)

# Load base model (full precision for merging)
print(f"\n1. Loading base model: {BASE_MODEL}")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

# Load LoRA adapters
print(f"\n2. Loading LoRA adapters from: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

# Merge and unload
print("\n3. Merging LoRA with base model...")
model = model.merge_and_unload()
print("Merge complete!")

# Save merged model
print(f"\n4. Saving merged model to: {MERGED_PATH}")
os.makedirs(MERGED_PATH, exist_ok=True)
model.save_pretrained(MERGED_PATH, safe_serialization=True)
tokenizer.save_pretrained(MERGED_PATH)
print("Saved!")

# Push to HuggingFace
print(f"\n5. Pushing to HuggingFace: {HF_REPO}")
model.push_to_hub(HF_REPO, safe_serialization=True)
tokenizer.push_to_hub(HF_REPO)

print("\n" + "="*60)
print("SUCCESS!")
print(f"Model available at: https://huggingface.co/{HF_REPO}")
print("="*60)
