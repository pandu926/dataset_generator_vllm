"""
Main Script: Synthetic Dataset Generator (Pipeline)
Reads raw JSON files -> Retrieves RAG Context -> Generates Multi-turn Conversations
"""

import os
import glob
import json
import random
from typing import List, Dict
from tqdm import tqdm

# Set seed for reproducibility
SEED = 758
random.seed(SEED)

# Import our custom modules
from src.llm_multiturn_generator import MultiTurnGenerator
from src.e5_embedding import create_embedding_service
try:
    from src.vllm_engine import VLLMEngine
    HAS_VLLM = True
except ImportError:
    print("Warning: vLLM not found. Running in simulation mode.")
    HAS_VLLM = False

# =============================================================================
# CONFIGURATION
# =============================================================================

RAW_DATA_DIR = "data/seeds" # Directory where dataset_*.json lives
OUTPUT_FILE = "data/raw/synthetic_multiturn_v1.json"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)  # Auto-create directory
TARGET_TOTAL = 2500  # Updated target
CHUNKS_PATH = "data/chunks/chunks.jsonl" # Path to RAG chunks

# Category distribution for balanced sampling (total = 2500)
# oot is limited to 5% as requested
CATEGORY_DISTRIBUTION = {
    "biaya": 400,      # 16%
    "beasiswa": 350,   # 14%
    "snk": 300,        # 12%
    "prodi": 300,      # 12%
    "umum": 275,       # 11%
    "fasilitas": 250,  # 10%
    "pofil": 250,      # 10%
    "alur": 250,       # 10%
    "oot": 125,        # 5%
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_seeds_by_category() -> Dict[str, List[Dict]]:
    """Load all questions from dataset_*.json grouped by category"""
    seeds_by_cat = {}
    files = glob.glob(os.path.join(RAW_DATA_DIR, "dataset_*.json"))
    
    print(f"Loading seeds from {len(files)} files...")
    for fpath in files:
        fname = os.path.basename(fpath)
        category = fname.replace("dataset_", "").replace(".json", "")
        
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            seeds_by_cat[category] = []
            for item in data:
                q = item.get("question") or item.get("Q")
                if q:
                    seeds_by_cat[category].append({
                        "seed_question": q,
                        "category": category,
                        "source_file": fname
                    })
            print(f"  - {fname}: {len(seeds_by_cat[category])} seeds")
        except Exception as e:
            print(f"  ! Error reading {fname}: {e}")
    
    total = sum(len(v) for v in seeds_by_cat.values())
    print(f"Total seeds loaded: {total}")
    return seeds_by_cat

def load_seeds_from_raw_json() -> List[Dict]:
    """Load all questions from dataset_*.json as seeds"""
    seeds = []
    files = glob.glob(os.path.join(RAW_DATA_DIR, "dataset_*.json"))
    
    print(f"Loading seeds from {len(files)} files...")
    for fpath in files:
        fname = os.path.basename(fpath)
        category = fname.replace("dataset_", "").replace(".json", "")
        
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            count = 0
            for item in data:
                # Extract question as seed
                q = item.get("question") or item.get("Q")
                if q:
                    seeds.append({
                        "seed_question": q,
                        "category": category,
                        "source_file": fname
                    })
                    count += 1
            print(f"  - {fname}: {count} seeds")
        except Exception as e:
            print(f"  ! Error reading {fname}: {e}")
            
    print(f"Total seeds loaded: {len(seeds)}")
    random.shuffle(seeds)
    return seeds

def load_rag_chunks() -> List[Dict]:
    """Load chunks for retrieval"""
    chunks = []
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                chunks.append(json.loads(line))
    else:
        print(f"Warning: Chunks file not found at {CHUNKS_PATH}")
    return chunks

def retrieve_context(seed: Dict, chunks: List[Dict], embed_service, top_k=3) -> str:
    """
    Semantic retrieval using E5 embeddings when available.
    Falls back to category-based filtering if embed_service is None.
    """
    query = seed["seed_question"]
    
    # Strategy 1: Use embedding service if available
    if embed_service is not None:
        try:
            # Encode query
            query_emb = embed_service.encode_query(query)
            
            # Get chunk contents and encode them
            chunk_contents = [c.get("content", "") for c in chunks]
            if chunk_contents:
                chunk_embs = embed_service.encode_passages(chunk_contents)
                
                # Find top-k similar chunks
                results = embed_service.find_similar(query_emb, chunk_embs, top_k=top_k, threshold=0.3)
                
                if results:
                    selected_chunks = [chunks[idx] for idx, _ in results]
                    return "\n\n".join([c.get("content", "") for c in selected_chunks])
        except Exception as e:
            print(f"Warning: Embedding retrieval failed, using fallback: {e}")
    
    # Strategy 2: Fallback - Filter by category then random sample
    relevant_chunks = [c for c in chunks if seed["category"] in c.get("id", "")]
    if not relevant_chunks:
        relevant_chunks = chunks  # Fallback to all if no category match
        
    if not relevant_chunks:
        return "Info tidak tersedia."
        
    # Pick random chunks from relevant set
    selected = random.sample(relevant_chunks, min(len(relevant_chunks), top_k))
    
    return "\n\n".join([c.get("content", "") for c in selected])

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("="*60)
    print(f"SYNTHETIC DATA GENERATION (SEED: {SEED})")
    print("="*60)
    
    # 1. Initialize Engine
    engine = None
    if HAS_VLLM:
        engine = VLLMEngine()
        print("vLLM Engine Ready.")
    
    # 2. Load Seeds by Category
    seeds_by_cat = load_seeds_by_category()
    if not seeds_by_cat:
        print("No seeds found!")
        return

    # 3. Load RAG Knowledge
    chunks = load_rag_chunks()
    
    # 4. Initialize Generator
    generator = MultiTurnGenerator(engine)
    
    # 5. Print distribution plan
    print("\n" + "="*60)
    print("CATEGORY DISTRIBUTION:")
    total_target = sum(CATEGORY_DISTRIBUTION.values())
    for cat, n in CATEGORY_DISTRIBUTION.items():
        pct = (n / total_target) * 100
        avail = len(seeds_by_cat.get(cat, []))
        print(f"  {cat:12}: {n:4} samples ({pct:5.1f}%) - Seeds available: {avail}")
    print(f"  {'TOTAL':12}: {total_target}")
    print("="*60 + "\n")
    
    # 6. BATCH Generation with Stratified Sampling
    BATCH_SIZE = 50  # Increased for faster generation
    generated_data = []
    category_counts = {cat: 0 for cat in CATEGORY_DISTRIBUTION}
    batch_num = 0
    
    pbar = tqdm(total=total_target, desc="Generating datasets", unit="conv")
    
    while sum(category_counts.values()) < total_target:
        batch_num += 1
        
        # Build batch with stratified sampling
        batch_chunks = []
        batch_seeds = []
        
        for cat, target in CATEGORY_DISTRIBUTION.items():
            if category_counts[cat] >= target:
                continue
            
            # Calculate how many to add from this category
            remaining_for_cat = target - category_counts[cat]
            slots_available = BATCH_SIZE - len(batch_chunks)
            
            if slots_available <= 0:
                break
            
            # Proportional allocation
            total_remaining = sum(CATEGORY_DISTRIBUTION[c] - category_counts[c] 
                                  for c in CATEGORY_DISTRIBUTION if category_counts[c] < CATEGORY_DISTRIBUTION[c])
            if total_remaining > 0:
                proportion = remaining_for_cat / total_remaining
                n_samples = max(1, min(int(slots_available * proportion) + 1, remaining_for_cat, slots_available))
            else:
                n_samples = 0
            
            # Get seeds for this category
            cat_seeds = seeds_by_cat.get(cat, [])
            if not cat_seeds:
                continue
            
            for _ in range(n_samples):
                if len(batch_chunks) >= BATCH_SIZE:
                    break
                
                # Cycle through seeds
                seed_idx = category_counts[cat] % len(cat_seeds)
                seed = cat_seeds[seed_idx]
                
                # Retrieve context
                context = retrieve_context(seed, chunks, None)
                
                chunk_input = {
                    "content": context,
                    "topic": cat,
                    "id": f"seed_{cat}_{category_counts[cat]}"
                }
                batch_chunks.append(chunk_input)
                batch_seeds.append((cat, seed))
        
        if not batch_chunks:
            break
        
        # Generate batch
        pbar.set_description(f"Batch {batch_num} ({len(batch_chunks)} prompts)")
        results = generator.generate_conversations_batch(batch_chunks)
        
        # Process results
        successful = 0
        for i, result in enumerate(results):
            if result:
                cat, seed = batch_seeds[i]
                
                item = {
                    "instruction": "Multi-turn conversation about UNSIQ",
                    "input": "",
                    "output": json.dumps(result["conversation"], ensure_ascii=False),
                    "text": "",
                    "category": cat,
                    "source": "synthetic_v2",
                    "metadata": result["metadata"]
                }
                generated_data.append(item)
                category_counts[cat] += 1
                successful += 1
        
        pbar.update(successful)
        
        # Checkpoint every 3 batches
        if batch_num % 3 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(generated_data, f, ensure_ascii=False, indent=2)
            tqdm.write(f"Checkpoint: {len(generated_data)} convs | " + 
                       " | ".join(f"{c}:{category_counts[c]}" for c in CATEGORY_DISTRIBUTION))
    
    pbar.close()
    
    # Final Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print(f"DONE! Generated {len(generated_data)} conversations.")
    print(f"Seed: {SEED}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("\nFinal Distribution:")
    for cat, n in category_counts.items():
        pct = (n / len(generated_data) * 100) if generated_data else 0
        print(f"  {cat:12}: {n:4} ({pct:5.1f}%)")
    print("="*60)


if __name__ == "__main__":
    main()

