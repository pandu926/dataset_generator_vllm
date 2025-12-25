"""
BERTScore Evaluation from Inference Results
Evaluate base_response and finetuned_response against expected
"""

import json
import argparse
import torch
from bert_score import score as bert_score

def evaluate_bertscore(input_path: str, output_path: str, model_type: str = "microsoft/deberta-xlarge-mnli", batch_size: int = 32):
    """Compute BERTScore for inference results."""
    
    print(f"\nLoading inference results from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    # Extract responses
    expected = [item["expected"] for item in data]
    base_responses = [item["base_response"] for item in data]
    finetuned_responses = [item["finetuned_response"] for item in data]
    
    # Filter empty responses
    def filter_empty(preds, refs):
        valid = [(p, r) for p, r in zip(preds, refs) if p and r]
        if not valid:
            return [], []
        return zip(*valid)
    
    print(f"\n{'='*60}")
    print("Computing BERTScore for BASE MODEL responses...")
    print(f"{'='*60}")
    
    base_preds, base_refs = filter_empty(base_responses, expected)
    base_preds, base_refs = list(base_preds), list(base_refs)
    
    P_base, R_base, F1_base = bert_score(
        base_preds, base_refs,
        model_type=model_type,
        batch_size=batch_size,
        verbose=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    base_scores = {
        "precision": float(P_base.mean()),
        "recall": float(R_base.mean()),
        "f1": float(F1_base.mean()),
        "precision_std": float(P_base.std()),
        "recall_std": float(R_base.std()),
        "f1_std": float(F1_base.std()),
        "num_samples": len(base_preds),
    }
    
    print(f"\nBase Model BERTScore F1: {base_scores['f1']:.4f}")
    
    print(f"\n{'='*60}")
    print("Computing BERTScore for FINETUNED MODEL responses...")
    print(f"{'='*60}")
    
    ft_preds, ft_refs = filter_empty(finetuned_responses, expected)
    ft_preds, ft_refs = list(ft_preds), list(ft_refs)
    
    P_ft, R_ft, F1_ft = bert_score(
        ft_preds, ft_refs,
        model_type=model_type,
        batch_size=batch_size,
        verbose=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    ft_scores = {
        "precision": float(P_ft.mean()),
        "recall": float(R_ft.mean()),
        "f1": float(F1_ft.mean()),
        "precision_std": float(P_ft.std()),
        "recall_std": float(R_ft.std()),
        "f1_std": float(F1_ft.std()),
        "num_samples": len(ft_preds),
    }
    
    print(f"\nFinetuned Model BERTScore F1: {ft_scores['f1']:.4f}")
    
    # Comparison
    improvement_f1 = ft_scores['f1'] - base_scores['f1']
    improvement_pct = (improvement_f1 / base_scores['f1']) * 100 if base_scores['f1'] > 0 else 0
    
    results = {
        "base_model": base_scores,
        "finetuned_model": ft_scores,
        "comparison": {
            "f1_improvement": improvement_f1,
            "f1_improvement_percent": improvement_pct,
            "precision_improvement": ft_scores['precision'] - base_scores['precision'],
            "recall_improvement": ft_scores['recall'] - base_scores['recall'],
        }
    }
    
    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*60)
    print("BERTSCORE EVALUATION RESULTS")
    print("="*60)
    print(f"\n{'Metric':<20} {'Base Model':<15} {'Fine-tuned':<15} {'Improvement':<15}")
    print("-"*65)
    print(f"{'Precision':<20} {base_scores['precision']:<15.4f} {ft_scores['precision']:<15.4f} {ft_scores['precision']-base_scores['precision']:+.4f}")
    print(f"{'Recall':<20} {base_scores['recall']:<15.4f} {ft_scores['recall']:<15.4f} {ft_scores['recall']-base_scores['recall']:+.4f}")
    print(f"{'F1 Score':<20} {base_scores['f1']:<15.4f} {ft_scores['f1']:<15.4f} {improvement_f1:+.4f} ({improvement_pct:+.1f}%)")
    print("="*60)
    print(f"\nResults saved to: {output_path}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERTScore Evaluation from Inference Results")
    parser.add_argument("--input", type=str, default="./outputs/inference_results_multiturn.json")
    parser.add_argument("--output", type=str, default="./outputs/bertscore_results.json")
    parser.add_argument("--model", type=str, default="microsoft/deberta-xlarge-mnli")
    parser.add_argument("--batch_size", type=int, default=32)
    
    args = parser.parse_args()
    
    evaluate_bertscore(args.input, args.output, args.model, args.batch_size)
