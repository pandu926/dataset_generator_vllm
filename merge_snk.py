import json

def merge_datasets(file1, file2, output_file):
    with open(file1, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)
    
    with open(file2, 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)
    
    merged_data = data1 + data2
    
    print(f"Data 1 ({file1}): {len(data1)} items")
    print(f"Data 2 ({file2}): {len(data2)} items")
    print(f"Merged Data: {len(merged_data)} items")
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(merged_data, f_out, indent=2, ensure_ascii=False)
    
    print(f"Saved merged dataset to: {output_file}")

if __name__ == "__main__":
    file_consecutive = "/workspace/dataset_generator_vllm/data/raw/categories/multiturn_snk_consecutive_model.json"
    file_normal = "/workspace/dataset_generator_vllm/data/raw/categories/multiturn_snk_normal.json"
    output_path = "/workspace/dataset_generator_vllm/data/raw/categories/multiturn_snk_clean_final.json"
    
    merge_datasets(file_normal, file_consecutive, output_path)
