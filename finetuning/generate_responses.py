"""
Single-turn Inference Script - First turn only
Output: id, category, prompt, expected, base_response, finetuned_response
"""

import os
import json
import torch
from typing import List, Dict
from dataclasses import dataclass
from tqdm import tqdm
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

@dataclass
class InferenceConfig:
    base_model_name: str = "google/gemma-3-1b-it"
    finetuned_model_path: str = "./outputs/gemma3-1b-qlora-no-cot/final_model"
    test_dataset_path: str = "../data/final/split/merged_all_categories_test_no_cot.json"
    output_path: str = "./outputs/inference_results_single_turn.json"
    batch_size: int = 32
    max_test_samples: int = 0
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95

def load_base_model(model_name: str, dtype: torch.dtype = torch.bfloat16):
    print(f"\nLoading base model: {model_name}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    
    print("Base model loaded!")
    return model, tokenizer

def load_finetuned_model(base_model_name: str, adapter_path: str, dtype: torch.dtype = torch.bfloat16):
    print(f"\nLoading fine-tuned model from: {adapter_path}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    
    model = PeftModel.from_pretrained(model, adapter_path)
    
    print("Fine-tuned model loaded!")
    return model, tokenizer

def load_test_data_single_turn(dataset_path: str, max_samples: int = 0) -> List[Dict]:
    """Load test data - FIRST TURN ONLY."""
    print(f"\nLoading test data from: {dataset_path}")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if max_samples and max_samples > 0 and len(data) > max_samples:
        test_data = data[:max_samples]
    else:
        test_data = data
    
    test_samples = []
    for item in test_data:
        conversation = item.get("conversation", [])
        if len(conversation) >= 2:
            # Get FIRST user message and FIRST model response only
            user_msg = None
            expected_response = None
            
            for turn in conversation:
                if turn["role"] == "user" and user_msg is None:
                    user_msg = turn["content"]
                elif turn["role"] == "model" and user_msg is not None and expected_response is None:
                    expected_response = turn["content"]
                    break  # Stop after first model response
            
            if user_msg and expected_response:
                test_samples.append({
                    "id": item.get("id", ""),
                    "prompt": user_msg,
                    "expected": expected_response,
                    "category": item.get("category", ""),
                })
    
    print(f"Loaded {len(test_samples)} test samples (single turn)")
    return test_samples

def generate_responses_batch(model, tokenizer, prompts: List[str], max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.95) -> List[str]:
    formatted_prompts = [
        f"<bos><start_of_turn>user\n{p}<end_of_turn>\n<start_of_turn>model\n"
        for p in prompts
    ]
    
    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    responses = []
    for output in outputs:
        full_response = tokenizer.decode(output, skip_special_tokens=False)
        
        if "<start_of_turn>model\n" in full_response:
            response = full_response.split("<start_of_turn>model\n")[-1]
            if "<end_of_turn>" in response:
                response = response.split("<end_of_turn>")[0]
            responses.append(response.strip())
        else:
            responses.append("")
    
    return responses

def generate_all_responses(model, tokenizer, test_samples: List[Dict], batch_size: int = 32, **gen_kwargs) -> List[str]:
    prompts = [s["prompt"] for s in test_samples]
    all_responses = []
    
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Generating responses"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]
        
        batch_responses = generate_responses_batch(
            model, tokenizer, batch_prompts, **gen_kwargs
        )
        all_responses.extend(batch_responses)
    
    return all_responses

def run_inference(config: InferenceConfig = None):
    if config is None:
        config = InferenceConfig()
    
    print("\n" + "="*60)
    print("SINGLE-TURN INFERENCE: BASE vs FINE-TUNED MODEL")
    print("="*60)
    
    test_samples = load_test_data_single_turn(config.test_dataset_path, config.max_test_samples)
    
    gen_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
    }
    
    # BASE model
    print("\n" + "-"*40)
    print("GENERATING BASE MODEL RESPONSES")
    print("-"*40)
    
    base_model, base_tokenizer = load_base_model(config.base_model_name)
    base_responses = generate_all_responses(
        base_model, base_tokenizer, test_samples,
        batch_size=config.batch_size, **gen_kwargs
    )
    
    del base_model
    torch.cuda.empty_cache()
    
    # FINETUNED model
    print("\n" + "-"*40)
    print("GENERATING FINE-TUNED MODEL RESPONSES")
    print("-"*40)
    
    ft_model, ft_tokenizer = load_finetuned_model(
        config.base_model_name, config.finetuned_model_path
    )
    ft_responses = generate_all_responses(
        ft_model, ft_tokenizer, test_samples,
        batch_size=config.batch_size, **gen_kwargs
    )
    
    del ft_model
    torch.cuda.empty_cache()
    
    # Build results
    results = []
    for i, sample in enumerate(test_samples):
        results.append({
            "id": sample.get("id", f"sample_{i}"),
            "category": sample.get("category", ""),
            "prompt": sample["prompt"],
            "expected": sample["expected"],
            "base_response": base_responses[i] if i < len(base_responses) else "",
            "finetuned_response": ft_responses[i] if i < len(ft_responses) else "",
        })
    
    # Save
    os.makedirs(os.path.dirname(config.output_path) if os.path.dirname(config.output_path) else ".", exist_ok=True)
    with open(config.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("SINGLE-TURN INFERENCE COMPLETE")
    print("="*60)
    print(f"Total samples: {len(results)}")
    print(f"Output saved to: {config.output_path}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-turn Inference")
    parser.add_argument("--base_model", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--finetuned_path", type=str, default="./outputs/gemma3-1b-qlora-no-cot/final_model")
    parser.add_argument("--test_dataset", type=str, default="../data/final/split/merged_all_categories_test_no_cot.json")
    parser.add_argument("--output", type=str, default="./outputs/inference_results_single_turn.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=0)
    
    args = parser.parse_args()
    
    config = InferenceConfig(
        base_model_name=args.base_model,
        finetuned_model_path=args.finetuned_path,
        test_dataset_path=args.test_dataset,
        output_path=args.output,
        batch_size=args.batch_size,
        max_test_samples=args.max_samples,
    )
    
    run_inference(config)
