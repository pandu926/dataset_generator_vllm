#!/usr/bin/env python3
"""
Clean and Merge Dataset
Menggabungkan semua file *_clean.json dari data/raw/categories,
menghapus field 'thought', dan menyimpan dalam format siap training.
"""

import os
import json
import glob
from typing import List, Dict

INPUT_DIR = "data/raw/categories"
OUTPUT_DIR = "data/merged"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "multiturn_merged_clean.json")


def remove_thought_from_conversation(conversation: List[Dict]) -> List[Dict]:
    """Remove 'thought' field from all messages and filter consecutive model messages."""
    cleaned = []
    for msg in conversation:
        # Skip messages that only contain thought (no actual content)
        if msg.get("role") == "model" and "thought" in msg and not msg.get("content"):
            continue
        
        # Remove thought field but keep the message
        clean_msg = {k: v for k, v in msg.items() if k != "thought"}
        
        # Skip empty content after removing thought
        if clean_msg.get("role") == "model" and clean_msg.get("content", "").startswith("I "):
            # This looks like a thought that was put in content field, check context
            if len(clean_msg.get("content", "")) < 500 and any(x in clean_msg.get("content", "") for x in ["I'll ", "I need to", "I will", "I must"]):
                continue
        
        cleaned.append(clean_msg)
    
    # Remove consecutive model messages (keep only the last one)
    final = []
    for i, msg in enumerate(cleaned):
        if i > 0 and msg.get("role") == "model" and cleaned[i-1].get("role") == "model":
            # Replace previous model message
            final[-1] = msg
        else:
            final.append(msg)
    
    return final


def rebuild_text_field(conversation: List[Dict]) -> str:
    """Rebuild text field from cleaned conversation."""
    parts = []
    for msg in conversation:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<start_of_turn>{role}\n{content}<end_of_turn>")
    return "\n".join(parts)


def parse_conversation_from_output(output_str: str) -> List[Dict]:
    """Parse conversation from JSON string in output field."""
    try:
        return json.loads(output_str)
    except json.JSONDecodeError:
        return []


def process_file(filepath: str) -> List[Dict]:
    """Process a single JSON file."""
    print(f"Processing: {filepath}")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  Error parsing {filepath}: {e}")
        return []
    
    if not data:
        print(f"  Empty file: {filepath}")
        return []
    
    cleaned_data = []
    for item in data:
        conv = None
        
        # Format 1: conversation field langsung (format baru)
        if "conversation" in item and isinstance(item.get("conversation"), list):
            conv = item["conversation"]
        # Format 2: conversation dalam field output sebagai JSON string (format lama)
        elif "output" in item and isinstance(item.get("output"), str):
            conv = parse_conversation_from_output(item["output"])
            if conv:
                # Hapus field lama yang tidak perlu
                item.pop("output", None)
                item.pop("instruction", None)
                item.pop("input", None)
        
        if conv:
            cleaned_conv = remove_thought_from_conversation(conv)
            
            # Skip if conversation became too short
            if len(cleaned_conv) < 2:
                continue
            
            item["conversation"] = cleaned_conv
            item["text"] = rebuild_text_field(cleaned_conv)
            item["num_turns"] = len(cleaned_conv)
            cleaned_data.append(item)
    
    print(f"  Cleaned {len(cleaned_data)} items from {os.path.basename(filepath)}")
    return cleaned_data


def main():
    print("="*60)
    print("CLEAN AND MERGE DATASET")
    print("="*60)
    
    # Find all *_clean.json files
    pattern = os.path.join(INPUT_DIR, "*_clean.json")
    files = glob.glob(pattern)
    
    # Also include multiturn_alur_pendaftaran_100.json if exists
    alur_file = os.path.join(INPUT_DIR, "multiturn_alur_pendaftaran_100.json")
    if os.path.exists(alur_file) and alur_file not in files:
        files.append(alur_file)
    
    print(f"Found {len(files)} files to process:")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    
    # Process all files
    all_data = []
    for filepath in files:
        file_data = process_file(filepath)
        all_data.extend(file_data)
    
    print(f"\nTotal items after merge: {len(all_data)}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save merged file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to: {OUTPUT_FILE}")
    
    # Split dataset 0.8 / 0.1 / 0.1
    import random
    random.seed(42)
    random.shuffle(all_data)
    
    total = len(all_data)
    train_size = int(total * 0.8)
    test_size = int(total * 0.1)
    
    train_data = all_data[:train_size]
    test_data = all_data[train_size:train_size + test_size]
    eval_data = all_data[train_size + test_size:]
    
    print(f"\nSplitting dataset (0.8/0.1/0.1):")
    print(f"  Train: {len(train_data)}")
    print(f"  Test:  {len(test_data)}")
    print(f"  Eval:  {len(eval_data)}")
    
    # Save split files
    train_file = os.path.join(OUTPUT_DIR, "multiturn_train.json")
    test_file = os.path.join(OUTPUT_DIR, "multiturn_test.json")
    eval_file = os.path.join(OUTPUT_DIR, "multiturn_eval.json")
    
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved to:")
    print(f"  {train_file}")
    print(f"  {test_file}")
    print(f"  {eval_file}")
    
    print("="*60)
    print("DONE!")


if __name__ == "__main__":
    main()
