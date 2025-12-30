import json
import os
# import pandas as pd
from collections import defaultdict

FILE_PATH = "/workspace/dataset_generator_vllm/finetuning/outputs/llm_judge_results.json"

def analyze_results():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    try:
        with open(FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return

    detailed = data.get("detailed_results", [])
    if not detailed:
        print("No detailed results found.")
        return

    # Aggregate scores by category
    stats = defaultdict(lambda: {"base_scores": [], "ft_scores": []})
    
    for item in detailed:
        category = item.get("category", "Uncategorized")
        base_score = item.get("base_model_scores", {}).get("total", 0)
        ft_score = item.get("finetuned_model_scores", {}).get("total", 0)
        
        stats[category]["base_scores"].append(base_score)
        stats[category]["ft_scores"].append(ft_score)

    print(f"{'Category':<20} | {'Count':<5} | {'Base Avg':<10} | {'FT Avg':<10} | {'Diff':<10} | {'Imp%':<10}")
    print("-" * 80)
    
    overall_base = []
    overall_ft = []

    for cat in sorted(stats.keys()):
        base_vals = stats[cat]["base_scores"]
        ft_vals = stats[cat]["ft_scores"]
        
        overall_base.extend(base_vals)
        overall_ft.extend(ft_vals)
        
        avg_base = sum(base_vals) / len(base_vals)
        avg_ft = sum(ft_vals) / len(ft_vals)
        diff = avg_ft - avg_base
        imp_pct = (diff / avg_base * 100) if avg_base > 0 else 0
        
        print(f"{cat:<20} | {len(base_vals):<5} | {avg_base:<10.2f} | {avg_ft:<10.2f} | {diff:<+10.2f} | {imp_pct:<+9.1f}%")

    print("-" * 80)
    if overall_base:
        avg_base_all = sum(overall_base) / len(overall_base)
        avg_ft_all = sum(overall_ft) / len(overall_ft)
        diff_all = avg_ft_all - avg_base_all
        imp_pct_all = (diff_all / avg_base_all * 100) if avg_base_all > 0 else 0
        print(f"{'OVERALL':<20} | {len(overall_base):<5} | {avg_base_all:<10.2f} | {avg_ft_all:<10.2f} | {diff_all:<+10.2f} | {imp_pct_all:<+9.1f}%")

if __name__ == "__main__":
    analyze_results()
