"""
Remove Thought from Dataset
Converts multiturn_dataset_cleaned.json to multiturn_dataset_cleaned_no_thought.json
by removing thought fields and <thought> tags from text.
"""

import json
import re
import os

INPUT_FILE = "../data/final/multiturn_dataset_cleaned.json"
OUTPUT_FILE = "../data/final/multiturn_dataset_cleaned_no_thought.json"

def remove_thought_from_dataset():
    print(f"Loading: {INPUT_FILE}")
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} items")
    
    cleaned_data = []
    
    for item in data:
        # Copy item
        new_item = item.copy()
        
        # Remove thought from conversation turns
        if "conversation" in new_item:
            new_conversation = []
            for turn in new_item["conversation"]:
                new_turn = {
                    "role": turn.get("role", ""),
                    "content": turn.get("content", "")
                }
                # Don't include thought field
                new_conversation.append(new_turn)
            new_item["conversation"] = new_conversation
        
        # Remove <thought>...</thought> from text field
        if "text" in new_item:
            text = new_item["text"]
            # Remove <thought>...</thought>\n pattern
            text = re.sub(r'<thought>.*?</thought>\n?', '', text, flags=re.DOTALL)
            new_item["text"] = text
        
        cleaned_data.append(new_item)
    
    # Save output
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nDONE!")
    print(f"  Processed: {len(cleaned_data)} items")
    print(f"  Saved to: {OUTPUT_FILE}")
    
    # Show sample
    if cleaned_data:
        print("\nSample (first item text):")
        print("-" * 40)
        print(cleaned_data[0].get("text", "")[:500])

if __name__ == "__main__":
    remove_thought_from_dataset()
