#!/usr/bin/env python3
"""
Single-Category Multi-turn Dataset Generator
Generates multi-turn conversations for ONE specific category only.
Usage: python generate_category.py --category alur_pendaftaran --multiplier 3
"""

import os
import json
import random
import argparse
from tqdm import tqdm

# Set seed for reproducibility
SEED = 758
random.seed(SEED)

# Import our custom modules
from src.llm_multiturn_generator import MultiTurnGenerator
try:
    from src.vllm_engine import VLLMEngine
    HAS_VLLM = True
except ImportError:
    print("Warning: vLLM not found. Running in simulation mode.")
    HAS_VLLM = False

# =============================================================================
# CONFIGURATION
# =============================================================================

RAW_DATA_DIR = "data/seeds"
OUTPUT_DIR = "data/raw/categories"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHUNKS_PATH = "data/chunks/chunks.jsonl"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_seeds_for_category(category: str):
    """Load seeds from a specific category file"""
    filepath = os.path.join(RAW_DATA_DIR, f"dataset_{category}.json")
    
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return []
    
    seeds = []
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for item in data:
        q = item.get("question") or item.get("Q")
        if q:
            seeds.append({
                "seed_question": q,
                "category": category,
                "source_file": os.path.basename(filepath)
            })
    
    print(f"Loaded {len(seeds)} seeds from {filepath}")
    return seeds


def load_rag_chunks():
    """Load chunks for retrieval"""
    chunks = []
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                chunks.append(json.loads(line))
        print(f"Loaded {len(chunks)} RAG chunks")
    else:
        print(f"Warning: Chunks file not found at {CHUNKS_PATH}")
    return chunks


def retrieve_context(seed, chunks, top_k=3):
    """Simple category-based context retrieval"""
    category = seed["category"]
    
    # Filter by category
    relevant_chunks = [c for c in chunks if category in c.get("id", "").lower()]
    if not relevant_chunks:
        relevant_chunks = chunks  # Fallback
    
    if not relevant_chunks:
        return "Info tidak tersedia."
    
    selected = random.sample(relevant_chunks, min(len(relevant_chunks), top_k))
    return "\n\n".join([c.get("content", "") for c in selected])


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate multi-turn dataset for a single category")
    parser.add_argument("--category", type=str, required=True, 
                        help="Category name (e.g., alur_pendaftaran, beasiswa)")
    parser.add_argument("--multiplier", type=int, default=3,
                        help="How many times to expand each seed (default: 3)")
    parser.add_argument("--batch_size", type=int, default=24,
                        help="Batch size for generation (default: 24)")
    args = parser.parse_args()
    
    category = args.category
    multiplier = args.multiplier
    batch_size = args.batch_size
    
    print("="*60)
    print(f"SINGLE-CATEGORY DATASET GENERATION")
    print(f"Category: {category}")
    print(f"Multiplier: {multiplier}x")
    print("="*60)
    
    # 1. Initialize Engine
    engine = None
    if HAS_VLLM:
        engine = VLLMEngine()
        print("vLLM Engine Ready.")
    
    # 2. Load Seeds for this category only
    seeds = load_seeds_for_category(category)
    if not seeds:
        print("No seeds found! Exiting.")
        return
    
    # 3. Load RAG Knowledge
    chunks = load_rag_chunks()
    
    # 4. Initialize Generator
    generator = MultiTurnGenerator(engine)
    
    # 5. Calculate target
    target = len(seeds) * multiplier
    print(f"\nTarget: {target} conversations ({len(seeds)} seeds x {multiplier})")
    
    # 6. BATCH Generation
    generated_data = []
    batch_num = 0
    
    # Create expanded seed list (each seed repeated 'multiplier' times)
    expanded_seeds = seeds * multiplier
    random.shuffle(expanded_seeds)
    
    pbar = tqdm(total=target, desc=f"Generating {category}", unit="conv")
    
    for i in range(0, len(expanded_seeds), batch_size):
        batch_num += 1
        batch_seeds = expanded_seeds[i:i+batch_size]
        
        # Prepare batch chunks
        batch_chunks = []
        for seed in batch_seeds:
            context = retrieve_context(seed, chunks)
            chunk_input = {
                "content": context,
                "topic": category,
                "id": f"seed_{category}_{len(batch_chunks)}"
            }
            batch_chunks.append(chunk_input)
        
        # Generate batch
        pbar.set_description(f"Batch {batch_num} ({len(batch_chunks)} prompts)")
        results = generator.generate_conversations_batch(batch_chunks)
        
        # Process results
        successful = 0
        for j, result in enumerate(results):
            if result:
                item = {
                    "instruction": f"Multi-turn conversation about UNSIQ - {category}",
                    "input": "",
                    "output": json.dumps(result["conversation"], ensure_ascii=False),
                    "text": "",
                    "category": category,
                    "source": "synthetic_category_v1",
                    "metadata": result["metadata"]
                }
                generated_data.append(item)
                successful += 1
        
        pbar.update(successful)
        
        # Checkpoint every 2 batches
        if batch_num % 2 == 0:
            output_file = os.path.join(OUTPUT_DIR, f"multiturn_{category}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(generated_data, f, ensure_ascii=False, indent=2)
            tqdm.write(f"Checkpoint: {len(generated_data)} conversations saved")
    
    pbar.close()
    
    # Final Save
    output_file = os.path.join(OUTPUT_DIR, f"multiturn_{category}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print(f"DONE! Generated {len(generated_data)} conversations for '{category}'")
    print(f"Success rate: {len(generated_data)/target*100:.1f}%")
    print(f"Saved to: {output_file}")
    print("="*60)
    
    # Print stats
    print("\nGenerator Stats:")
    print(f"  Generated: {generator.stats['generated']}")
    print(f"  Failed Validation: {generator.stats['failed_validation']}")


if __name__ == "__main__":
    main()
