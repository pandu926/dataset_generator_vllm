import json

def check_balance(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    balanced_count = 0
    unbalanced_count = 0
    unbalanced_items = []

    for item in data:
        conversation = item.get('conversation', [])
        user_roles = [msg['role'] for msg in conversation if msg['role'] == 'user']
        model_roles = [msg['role'] for msg in conversation if msg['role'] == 'model']

        if len(user_roles) == len(model_roles):
            balanced_count += 1
        else:
            unbalanced_count += 1
            unbalanced_items.append({
                'id': item.get('id'),
                'user_count': len(user_roles),
                'model_count': len(model_roles),
                'conversation': conversation
            })

    print(f"Total items: {len(data)}")
    print(f"Balanced items: {balanced_count}")
    print(f"Unbalanced items: {unbalanced_count}")

    if unbalanced_items:
        print("\nExamples of unbalanced items:")
        for ui in unbalanced_items[:5]: # Show first 5
            print(f"ID: {ui['id']} (User: {ui['user_count']}, Model: {ui['model_count']})")
            for msg in ui['conversation']:
                 print(f"  {msg['role']}: {msg['content'][:100]}...") # Print start of content
            print("-" * 20)
            
    # Save unbalanced IDs if needed or detailed report
    # with open('unbalanced_report.json', 'w') as f:
    #     json.dump(unbalanced_items, f, indent=2)

if __name__ == "__main__":
    check_balance('/workspace/dataset_generator_vllm/data/raw/categories/multiturn_snk_consecutive_model.json')
