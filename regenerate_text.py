import json

def regenerate_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    count = 0
    for item in data:
        conversation = item.get('conversation', [])
        if not conversation:
            continue
            
        text = ""
        for msg in conversation:
            role = msg.get('role', '')
            content = msg.get('content', '')
            text += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
            
        item['text'] = text
        count += 1
            
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"Regenerated 'text' field for {count} items in {file_path}")

if __name__ == "__main__":
    # Process both files just to be sure, or just the requested one. 
    # User specifically asked for normal.json
    regenerate_text('/workspace/dataset_generator_vllm/data/raw/categories/multiturn_snk_normal.json')
