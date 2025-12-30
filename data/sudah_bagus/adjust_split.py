import json

TRAIN_FILE = "/workspace/dataset_generator_vllm/data/sudah_bagus/train.json"
TEST_FILE = "/workspace/dataset_generator_vllm/data/sudah_bagus/test.json"

def main():
    # Load files
    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"Before: Train={len(train_data)}, Test={len(test_data)}")
    
    # Group train by category
    train_by_category = {}
    for item in train_data:
        cat = item.get('category', 'uncategorized')
        if cat not in train_by_category:
            train_by_category[cat] = []
        train_by_category[cat].append(item)
    
    # We need test to have 220 items. Currently 206, need 14 more.
    # Move 1 from each category until we reach 220
    target_test_count = 220
    items_to_move = target_test_count - len(test_data)
    
    print(f"Need to move {items_to_move} items from train to test")
    
    moved = 0
    categories = list(train_by_category.keys())
    
    # Keep cycling through categories until we've moved enough
    cat_idx = 0
    while moved < items_to_move:
        cat = categories[cat_idx % len(categories)]
        if len(train_by_category[cat]) > 0:
            item = train_by_category[cat].pop(0)
            test_data.append(item)
            moved += 1
            print(f"  Moved 1 from '{cat}' -> Test (total moved: {moved})")
        cat_idx += 1
        
        # Safety check to avoid infinite loop
        if cat_idx > items_to_move * len(categories):
            break
    
    # Rebuild train_data from the remaining items
    new_train_data = []
    for cat, items in train_by_category.items():
        new_train_data.extend(items)
    
    print(f"After: Train={len(new_train_data)}, Test={len(test_data)}")
    
    # Save files
    with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
        json.dump(new_train_data, f, indent=2, ensure_ascii=False)
    with open(TEST_FILE, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print("Done!")

if __name__ == "__main__":
    main()
