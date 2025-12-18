"""
BERTScore Evaluation Script
Compare base model vs fine-tuned model performance using batch processing
Optimized for A100 80GB
"""

import os
import json
import torch
from typing import List, Dict, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from bert_score import score as bert_score
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EvalConfig:
    """Evaluation configuration."""
    base_model_name: str = "google/gemma-3-1b-it"
    finetuned_model_path: str = "./outputs/gemma3-1b-qlora-sft/final_model"
    test_dataset_path: str = "../data/final/multiturn_dataset_cleaned.json"
    output_path: str = "./outputs/evaluation_results.json"
    
    # Batch processing
    batch_size: int = 16  # Generate responses in batches
    max_test_samples: int = 100  # Number of samples to evaluate
    
    # Generation parameters
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    
    # BERTScore
    bertscore_model: str = "microsoft/deberta-xlarge-mnli"  # Best for semantic similarity
    bertscore_batch_size: int = 32

# =============================================================================
# MODEL LOADING
# =============================================================================

def load_base_model(model_name: str, dtype: torch.dtype = torch.bfloat16):
    """Load the base model (not fine-tuned) with 4-bit quantization."""
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
    tokenizer.padding_side = "left"  # Left padding for batch generation
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    
    print("Base model loaded!")
    return model, tokenizer

def load_finetuned_model(
    base_model_name: str, 
    adapter_path: str, 
    dtype: torch.dtype = torch.bfloat16
):
    """Load the fine-tuned model with LoRA adapters."""
    print(f"\nLoading fine-tuned model from: {adapter_path}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load tokenizer from adapter path (has correct settings)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Left padding for batch generation
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(model, adapter_path)
    
    print("Fine-tuned model loaded!")
    return model, tokenizer

# =============================================================================
# DATA LOADING
# =============================================================================

def load_test_data(dataset_path: str, max_samples: int = 100) -> List[Dict]:
    """Load test data and extract prompts with expected responses."""
    print(f"\nLoading test data from: {dataset_path}")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Take last N samples as test set (or random sample)
    test_data = data[-max_samples:] if len(data) > max_samples else data
    
    # Extract first user message and expected model response from each conversation
    test_samples = []
    for item in test_data:
        conversation = item.get("conversation", [])
        if len(conversation) >= 2:
            # Get first user message
            user_msg = None
            expected_response = None
            
            for i, turn in enumerate(conversation):
                if turn["role"] == "user" and user_msg is None:
                    user_msg = turn["content"]
                elif turn["role"] == "model" and user_msg is not None and expected_response is None:
                    expected_response = turn["content"]
                    break
            
            if user_msg and expected_response:
                test_samples.append({
                    "id": item.get("id", ""),
                    "prompt": user_msg,
                    "expected": expected_response,
                    "category": item.get("category", ""),
                })
    
    print(f"Loaded {len(test_samples)} test samples")
    return test_samples

# =============================================================================
# BATCH GENERATION
# =============================================================================

def generate_responses_batch(
    model, 
    tokenizer, 
    prompts: List[str],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> List[str]:
    """Generate responses for a batch of prompts."""
    
    # Format prompts for Gemma
    formatted_prompts = [
        f"<bos><start_of_turn>user\n{p}<end_of_turn>\n<start_of_turn>model\n"
        for p in prompts
    ]
    
    # Tokenize batch
    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode responses
    responses = []
    for i, output in enumerate(outputs):
        full_response = tokenizer.decode(output, skip_special_tokens=False)
        
        # Extract model response
        if "<start_of_turn>model\n" in full_response:
            response = full_response.split("<start_of_turn>model\n")[-1]
            if "<end_of_turn>" in response:
                response = response.split("<end_of_turn>")[0]
            responses.append(response.strip())
        else:
            responses.append("")
    
    return responses

def generate_all_responses(
    model, 
    tokenizer, 
    test_samples: List[Dict],
    batch_size: int = 16,
    **gen_kwargs
) -> List[str]:
    """Generate responses for all test samples using batch processing."""
    
    prompts = [s["prompt"] for s in test_samples]
    all_responses = []
    
    # Process in batches
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

# =============================================================================
# BERTSCORE EVALUATION
# =============================================================================

def compute_bertscore_batch(
    predictions: List[str],
    references: List[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    batch_size: int = 32,
) -> Dict[str, float]:
    """Compute BERTScore for predictions vs references using batch processing."""
    
    print(f"\nComputing BERTScore (model: {model_type})...")
    print(f"Batch size: {batch_size}")
    
    # Filter empty predictions
    valid_pairs = [
        (p, r) for p, r in zip(predictions, references) 
        if p and r
    ]
    
    if not valid_pairs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    preds, refs = zip(*valid_pairs)
    
    # Compute BERTScore with batch processing
    P, R, F1 = bert_score(
        list(preds),
        list(refs),
        model_type=model_type,
        batch_size=batch_size,
        verbose=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    return {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F1.mean()),
        "precision_std": float(P.std()),
        "recall_std": float(R.std()),
        "f1_std": float(F1.std()),
        "num_samples": len(valid_pairs),
    }

# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate(config: EvalConfig = None):
    """Main evaluation function comparing base vs fine-tuned model."""
    
    if config is None:
        config = EvalConfig()
    
    print("\n" + "="*60)
    print("BERTSCORE EVALUATION: BASE vs FINE-TUNED MODEL")
    print("="*60)
    
    # Load test data
    test_samples = load_test_data(config.test_dataset_path, config.max_test_samples)
    expected_responses = [s["expected"] for s in test_samples]
    
    results = {
        "config": {
            "base_model": config.base_model_name,
            "finetuned_model": config.finetuned_model_path,
            "num_samples": len(test_samples),
            "batch_size": config.batch_size,
            "bertscore_model": config.bertscore_model,
        },
        "base_model": {},
        "finetuned_model": {},
        "comparison": {},
    }
    
    gen_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
    }
    
    # ===========================================
    # Evaluate BASE model
    # ===========================================
    print("\n" + "-"*40)
    print("EVALUATING BASE MODEL")
    print("-"*40)
    
    base_model, base_tokenizer = load_base_model(config.base_model_name)
    
    base_responses = generate_all_responses(
        base_model, base_tokenizer, test_samples,
        batch_size=config.batch_size, **gen_kwargs
    )
    
    # Free memory
    del base_model
    torch.cuda.empty_cache()
    
    base_scores = compute_bertscore_batch(
        base_responses, expected_responses,
        model_type=config.bertscore_model,
        batch_size=config.bertscore_batch_size,
    )
    
    results["base_model"] = {
        "bertscore": base_scores,
        "sample_responses": base_responses[:5],  # Save first 5 for inspection
    }
    
    print(f"\nBase Model BERTScore F1: {base_scores['f1']:.4f}")
    
    # ===========================================
    # Evaluate FINE-TUNED model
    # ===========================================
    print("\n" + "-"*40)
    print("EVALUATING FINE-TUNED MODEL")
    print("-"*40)
    
    ft_model, ft_tokenizer = load_finetuned_model(
        config.base_model_name, config.finetuned_model_path
    )
    
    ft_responses = generate_all_responses(
        ft_model, ft_tokenizer, test_samples,
        batch_size=config.batch_size, **gen_kwargs
    )
    
    # Free memory
    del ft_model
    torch.cuda.empty_cache()
    
    ft_scores = compute_bertscore_batch(
        ft_responses, expected_responses,
        model_type=config.bertscore_model,
        batch_size=config.bertscore_batch_size,
    )
    
    results["finetuned_model"] = {
        "bertscore": ft_scores,
        "sample_responses": ft_responses[:5],
    }
    
    print(f"\nFine-tuned Model BERTScore F1: {ft_scores['f1']:.4f}")
    
    # ===========================================
    # COMPARISON
    # ===========================================
    improvement_f1 = ft_scores['f1'] - base_scores['f1']
    improvement_pct = (improvement_f1 / base_scores['f1']) * 100 if base_scores['f1'] > 0 else 0
    
    results["comparison"] = {
        "f1_improvement": improvement_f1,
        "f1_improvement_percent": improvement_pct,
        "precision_improvement": ft_scores['precision'] - base_scores['precision'],
        "recall_improvement": ft_scores['recall'] - base_scores['recall'],
    }
    
    # ===========================================
    # SAVE RESULTS
    # ===========================================
    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    with open(config.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\n{'Metric':<20} {'Base Model':<15} {'Fine-tuned':<15} {'Improvement':<15}")
    print("-"*65)
    print(f"{'Precision':<20} {base_scores['precision']:<15.4f} {ft_scores['precision']:<15.4f} {ft_scores['precision']-base_scores['precision']:+.4f}")
    print(f"{'Recall':<20} {base_scores['recall']:<15.4f} {ft_scores['recall']:<15.4f} {ft_scores['recall']-base_scores['recall']:+.4f}")
    print(f"{'F1 Score':<20} {base_scores['f1']:<15.4f} {ft_scores['f1']:<15.4f} {improvement_f1:+.4f} ({improvement_pct:+.1f}%)")
    print("="*60)
    print(f"\nResults saved to: {config.output_path}")
    
    return results

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERTScore Evaluation")
    parser.add_argument("--base_model", type=str, default="google/gemma-3-1b-it",
                        help="Base model name")
    parser.add_argument("--finetuned_path", type=str, 
                        default="./outputs/gemma3-1b-qlora-sft/final_model",
                        help="Path to fine-tuned model")
    parser.add_argument("--test_dataset", type=str,
                        default="../data/final/multiturn_dataset_cleaned.json",
                        help="Path to test dataset")
    parser.add_argument("--output", type=str, default="./outputs/evaluation_results.json",
                        help="Path to save results")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for generation")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Maximum number of test samples")
    parser.add_argument("--bertscore_model", type=str, 
                        default="microsoft/deberta-xlarge-mnli",
                        help="Model for BERTScore")
    
    args = parser.parse_args()
    
    config = EvalConfig(
        base_model_name=args.base_model,
        finetuned_model_path=args.finetuned_path,
        test_dataset_path=args.test_dataset,
        output_path=args.output,
        batch_size=args.batch_size,
        max_test_samples=args.max_samples,
        bertscore_model=args.bertscore_model,
    )
    
    evaluate(config)
