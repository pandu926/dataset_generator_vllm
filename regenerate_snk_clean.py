#!/usr/bin/env python3
"""
Regenerate multiturn_snk_clean.json from multiturn_snk_300.json
Properly removes thinking content from model responses.
"""

import os
import json
from typing import List, Dict

INPUT_FILE = "data/raw/categories/multiturn_snk_300.json"
OUTPUT_FILE = "data/raw/categories/multiturn_snk_clean.json"


def is_thinking_content(content: str) -> bool:
    """Check if content appears to be thinking/instruction rather than actual response."""
    if not content:
        return False
    
    content_lower = content.lower().strip()
    
    # Pattern 1: Starts with "I will/I'll/I need to/I must" (English thinking)
    english_thinking_starters = [
        "i will ", "i'll ", "i need to", "i must ", "i should ",
        "i'm going to", "i am going to", "let me ", "i have to "
    ]
    
    # Pattern 2: Imperative/instruction-like phrases (LLM self-instructions)
    instruction_patterns = [
        "inform the user", "provide the information", "provide information",
        "briefly explain", "explain the", "state that", "state the",
        "confirm the user", "answer the", "answer:", "respond with",
        "present the", "list the", "give the", "tell the user",
        "acknowledge", "direct the user", "direct them",
        "a polite", "a formal", "a courteous", "reiterate the",
        "provide a concise", "provide the relevant",
        "process verification requires"  # Mixed English/Indonesian thinking
    ]
    
    # Pattern 3: Numbered analysis format (1. Analyze: 2. Retrieve: 3. Answer:)
    analysis_patterns = [
        "1. analyze:", "2. retrieve:", "3. answer:",
        "analyze:", "retrieve:", "the context",
        "based on the context", "according to the context"
    ]
    
    # Check for short instruction-like content
    if len(content) < 300:
        # Check English thinking starters
        for pattern in english_thinking_starters:
            if content_lower.startswith(pattern):
                return True
        
        # Check instruction patterns
        for pattern in instruction_patterns:
            if pattern in content_lower:
                return True
    
    # Check analysis patterns regardless of length
    for pattern in analysis_patterns:
        if pattern in content_lower:
            return True
    
    return False


def remove_thought_from_conversation(conversation: List[Dict]) -> List[Dict]:
    """Remove 'thought' field from all messages and filter thinking content."""
    cleaned = []
    for msg in conversation:
        # Skip messages that only contain thought (no actual content)
        if msg.get("role") == "model" and "thought" in msg and not msg.get("content"):
            continue
        
        # Remove thought field but keep the message
        clean_msg = {k: v for k, v in msg.items() if k != "thought"}
        
        # Skip if content is actually thinking/instruction
        if clean_msg.get("role") == "model":
            content = clean_msg.get("content", "")
            if is_thinking_content(content):
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


def main():
    print("=" * 60)
    print("REGENERATE SNK CLEAN DATASET")
    print("=" * 60)
    
    # Load raw data
    print(f"\nLoading: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} items")
    
    cleaned_data = []
    skipped_items = []
    
    for item in data:
        conv = None
        item_id = item.get("id", "unknown")
        
        # Format 1: conversation field langsung (format baru)
        if "conversation" in item and isinstance(item.get("conversation"), list):
            conv = item["conversation"]
        # Format 2: conversation dalam field output sebagai JSON string (format lama)
        elif "output" in item and isinstance(item.get("output"), str):
            conv = parse_conversation_from_output(item["output"])
        
        if conv:
            cleaned_conv = remove_thought_from_conversation(conv)
            
            # Check if conversation became too short
            if len(cleaned_conv) < 2:
                skipped_items.append(item_id)
                continue
            
            # Create clean item
            clean_item = {
                "id": item.get("id"),
                "source": item.get("source", "synthetic_snk_v1"),
                "category": item.get("category", "snk"),
                "subcategory": item.get("subcategory"),
                "persona": item.get("persona"),
                "complexity": item.get("complexity"),
                "conversation": cleaned_conv,
                "text": rebuild_text_field(cleaned_conv),
                "num_turns": len(cleaned_conv)
            }
            cleaned_data.append(clean_item)
    
    print(f"\nCleaned: {len(cleaned_data)} items")
    if skipped_items:
        print(f"Skipped {len(skipped_items)} items (too short after cleaning):")
        for sid in skipped_items[:5]:
            print(f"  - {sid}")
        if len(skipped_items) > 5:
            print(f"  ... and {len(skipped_items) - 5} more")
    
    # Sample validation - check for thinking content
    print("\n--- SAMPLE VALIDATION ---")
    issues_found = 0
    for item in cleaned_data[:20]:
        for msg in item.get("conversation", []):
            if msg.get("role") == "model":
                content = msg.get("content", "")
                if is_thinking_content(content):
                    print(f"ISSUE in {item['id']}: {content[:100]}...")
                    issues_found += 1
    
    if issues_found == 0:
        print("✓ No thinking content found in first 20 items")
    else:
        print(f"⚠ Found {issues_found} potential issues")
    
    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Saved to: {OUTPUT_FILE}")
    print(f"  Total items: {len(cleaned_data)}")
    
    # Show sample clean item
    if cleaned_data:
        sample = cleaned_data[0]
        print(f"\n--- SAMPLE CLEAN ITEM ({sample['id']}) ---")
        for i, msg in enumerate(sample["conversation"][:4]):
            print(f"  [{msg['role']}]: {msg['content'][:80]}...")
    
    print("\n" + "=" * 60)
    print("DONE!")


if __name__ == "__main__":
    main()
