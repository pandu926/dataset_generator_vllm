"""
Dataset Formatter: Clean and format synthetic_multiturn_v1.json
Converts to proper multi-turn format according to template.
"""

import json
import os
from typing import List, Dict

INPUT_FILE = "data/raw/synthetic_multiturn_v1.json"
OUTPUT_FILE = "data/final/multiturn_dataset_cleaned.json"

# Create output directory
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def format_conversation_to_text(conversation: List[Dict]) -> str:
    """Convert conversation list to formatted text string for training."""
    formatted_turns = []
    
    for turn in conversation:
        role = turn.get("role", "")
        content = turn.get("content", "")
        thought = turn.get("thought", "")
        
        if role == "user":
            formatted_turns.append(f"<start_of_turn>user\n{content}<end_of_turn>")
        elif role == "model":
            # Include thought in training if present (implicit CoT)
            if thought:
                model_response = f"<thought>{thought}</thought>\n{content}"
            else:
                model_response = content
            formatted_turns.append(f"<start_of_turn>model\n{model_response}<end_of_turn>")
    
    return "\n".join(formatted_turns)

def clean_and_format_dataset():
    """Main function to clean and reformat the dataset."""
    print(f"Loading: {INPUT_FILE}")
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    print(f"Loaded {len(raw_data)} items")
    
    cleaned_data = []
    failed = 0
    
    for i, item in enumerate(raw_data):
        try:
            # Parse the conversation from JSON string
            output_str = item.get("output", "[]")
            conversation = json.loads(output_str)
            
            if not isinstance(conversation, list) or len(conversation) < 2:
                failed += 1
                continue
            
            # Normalize conversation structure - ensure consistent fields
            normalized_conversation = []
            for turn in conversation:
                role = turn.get("role", "")
                content = turn.get("content", "").strip()
                thought = turn.get("thought", "").strip()
                
                # Fix: If model content is empty but thought contains "3. Answer:", extract it
                if role == "model" and not content and thought:
                    # Try to extract answer from thought
                    if "3. Answer:" in thought:
                        parts = thought.split("3. Answer:", 1)
                        reasoning = parts[0].strip()  # Keep 1. Analyze, 2. Retrieve
                        answer = parts[1].strip() if len(parts) > 1 else ""
                        thought = reasoning
                        content = answer
                    elif "Answer:" in thought:
                        parts = thought.split("Answer:", 1)
                        reasoning = parts[0].strip()
                        answer = parts[1].strip() if len(parts) > 1 else ""
                        thought = reasoning
                        content = answer
                
                # All turns have: role, thought, content (consistent structure)
                normalized_turn = {
                    "role": role,
                    "thought": thought,  # Will be "" for user turns
                    "content": content
                }
                normalized_conversation.append(normalized_turn)
            
            # Format text for training
            formatted_text = format_conversation_to_text(normalized_conversation)
            
            # Build cleaned item
            cleaned_item = {
                "id": f"synth_{i:05d}",
                "source": item.get("source", "synthetic_v1"),
                "category": item.get("category", "umum"),
                "persona": item.get("metadata", {}).get("persona", ""),
                "complexity": item.get("metadata", {}).get("complexity", ""),
                "conversation": normalized_conversation,  # Normalized structure
                "text": formatted_text,  # Training-ready text
                "num_turns": len(conversation)
            }
            
            cleaned_data.append(cleaned_item)
            
        except Exception as e:
            failed += 1
            print(f"Error at item {i}: {e}")
    
    # Save cleaned dataset
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"DONE!")
    print(f"  Success: {len(cleaned_data)}")
    print(f"  Failed: {failed}")
    print(f"  Saved to: {OUTPUT_FILE}")
    print(f"{'='*60}")
    
    # Show sample
    if cleaned_data:
        print("\nSample formatted output:")
        print("-"*40)
        print(cleaned_data[0]["text"][:1500])

if __name__ == "__main__":
    clean_and_format_dataset()
