import json
import os

FILES = [
    "/workspace/dataset_generator_vllm/data/sudah_bagus/train.json",
    "/workspace/dataset_generator_vllm/data/sudah_bagus/test.json",
    "/workspace/dataset_generator_vllm/data/sudah_bagus/eval.json"
]

def regenerate_text(conversation):
    """
    Regenerate text field from conversation list.
    Format:
    <start_of_turn>user
    [content]<end_of_turn>
    <start_of_turn>model
    [content]<end_of_turn>
    """
    text_parts = []
    for turn in conversation:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        text_parts.append(f"<start_of_turn>{role}\n{content}<end_of_turn>")
    return "\n".join(text_parts)

def process_file(filepath):
    print(f"Processing {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    count = 0
    for item in data:
        if 'conversation' in item:
            item['text'] = regenerate_text(item['conversation'])
            count += 1
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"  Regenerated {count} text fields.")

def main():
    print("Starting text regeneration...")
    for filepath in FILES:
        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"File not found: {filepath}")
    print("Done!")

if __name__ == "__main__":
    main()
