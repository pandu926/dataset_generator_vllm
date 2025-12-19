"""
Dataset Splitter for Training Pipeline
Split dataset into train/eval/test sets (80/10/10 default)
"""

import os
import json
import random
import argparse
from typing import Tuple, List, Dict


def split_dataset(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.75,
    eval_ratio: float = 0.15,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[str, str, str]:
    """
    Split dataset into train/eval/test sets.
    
    Args:
        input_path: Path to input JSON file
        output_dir: Directory to save split files
        train_ratio: Ratio for training set (default: 0.75)
        eval_ratio: Ratio for evaluation set (default: 0.1)
        test_ratio: Ratio for test set (default: 0.1)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_path, eval_path, test_path)
    """
    # Validate ratios
    assert abs(train_ratio + eval_ratio + test_ratio - 1.0) < 0.001, \
        f"Ratios must sum to 1.0, got {train_ratio + eval_ratio + test_ratio}"
    
    # Load dataset
    print(f"Loading dataset from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    total = len(data)
    print(f"Total samples: {total}")
    
    # Shuffle with seed for reproducibility
    random.seed(seed)
    random.shuffle(data)
    
    # Calculate split indices
    train_end = int(total * train_ratio)
    eval_end = train_end + int(total * eval_ratio)
    
    # Split data
    train_data = data[:train_end]
    eval_data = data[train_end:eval_end]
    test_data = data[eval_end:]
    
    print(f"Split results:")
    print(f"  Train: {len(train_data)} samples ({len(train_data)/total*100:.1f}%)")
    print(f"  Eval:  {len(eval_data)} samples ({len(eval_data)/total*100:.1f}%)")
    print(f"  Test:  {len(test_data)} samples ({len(test_data)/total*100:.1f}%)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Save split files
    train_path = os.path.join(output_dir, f"{base_name}_train.json")
    eval_path = os.path.join(output_dir, f"{base_name}_eval.json")
    test_path = os.path.join(output_dir, f"{base_name}_test.json")
    
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"Saved train set to: {train_path}")
    
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    print(f"Saved eval set to: {eval_path}")
    
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"Saved test set to: {test_path}")
    
    # Save split info
    info_path = os.path.join(output_dir, f"{base_name}_split_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump({
            "source": input_path,
            "total_samples": total,
            "train_samples": len(train_data),
            "eval_samples": len(eval_data),
            "test_samples": len(test_data),
            "train_ratio": train_ratio,
            "eval_ratio": eval_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "train_path": train_path,
            "eval_path": eval_path,
            "test_path": test_path,
        }, f, indent=2)
    
    return train_path, eval_path, test_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/eval/test")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input dataset JSON")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save split files")
    parser.add_argument("--train_ratio", type=float, default=0.75,
                        help="Training set ratio (default: 0.75)")
    parser.add_argument("--eval_ratio", type=float, default=0.15,
                        help="Evaluation set ratio (default: 0.15)")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Test set ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    split_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
