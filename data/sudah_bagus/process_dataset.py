import json
import os
import random
import glob

# Set random seed for reproducibility
random.seed(42)

SOURCE_DIR = "/workspace/dataset_generator_vllm/data/sudah_bagus"
OUTPUT_FILES = {
    "train": os.path.join(SOURCE_DIR, "train.json"),
    "test": os.path.join(SOURCE_DIR, "test.json"),
    "eval": os.path.join(SOURCE_DIR, "eval.json")
}

def load_all_data(source_dir):
    all_data = []
    files = glob.glob(os.path.join(source_dir, "*.json"))
    
    # Exclude output files if they already exist to avoid reading them recursively
    exclude_names = ["train.json", "test.json", "eval.json"]
    files = [f for f in files if os.path.basename(f) not in exclude_names]
    
    print(f"Found {len(files)} files to process.")
    
    for file_path in files:
        print(f"Reading {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    print(f"Warning: {file_path} does not contain a list of objects. Skipping.")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    return all_data

def split_data(all_data):
    # Group by category
    grouped_data = {}
    for item in all_data:
        # Use 'category' field, default to 'uncategorized' if missing
        category = item.get('category', 'uncategorized')
        if category not in grouped_data:
            grouped_data[category] = []
        grouped_data[category].append(item)
    
    print(f"Found categories: {list(grouped_data.keys())}")
    
    train_set = []
    test_set = []
    eval_set = []
    
    for category, items in grouped_data.items():
        print(f"Processing category '{category}': {len(items)} items")
        random.shuffle(items)
        
        n = len(items)
        n_train = int(n * 0.8)
        n_test = int(n * 0.1)
        # Remaining goes to eval to ensure sum is n
        
        train_chunk = items[:n_train]
        test_chunk = items[n_train:n_train + n_test]
        eval_chunk = items[n_train + n_test:]
        
        train_set.extend(train_chunk)
        test_set.extend(test_chunk)
        eval_set.extend(eval_chunk)
        
        print(f"  Split: Train={len(train_chunk)}, Test={len(test_chunk)}, Eval={len(eval_chunk)}")
        
    return train_set, test_set, eval_set

def save_json(data, filepath):
    print(f"Saving {len(data)} items to {filepath}...")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    print("Starting dataset processing...")
    all_data = load_all_data(SOURCE_DIR)
    print(f"Total items loaded: {len(all_data)}")
    
    if len(all_data) == 0:
        print("No data found! Exiting.")
        return

    train, test, eval_ = split_data(all_data)
    
    print(f"Total Split: Train={len(train)}, Test={len(test)}, Eval={len(eval_)}")
    
    save_json(train, OUTPUT_FILES["train"])
    save_json(test, OUTPUT_FILES["test"])
    save_json(eval_, OUTPUT_FILES["eval"])
    
    print("Done!")

if __name__ == "__main__":
    main()
