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
TARGET_TOTAL = 1000
CHUNKS_PATH = "data/chunks/chunks.jsonl" # Path to RAG chunks

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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
    print("STARTING SYNTHETIC DATA GENERATION")
    print("="*60)
    
    # 1. Initialize Engine
    engine = None
    if HAS_VLLM:
        engine = VLLMEngine()
        print("vLLM Engine Ready.")
    
    # 2. Load Seeds
    seeds = load_seeds_from_raw_json()
    if not seeds:
        print("No seeds found!")
        return

    # 3. Load RAG Knowledge
    chunks = load_rag_chunks()
    
    # 4. Initialize Generator
    generator = MultiTurnGenerator(engine)
    
    # 5. Generation Loop
    generated_data = []
    
    # Calculate target per seed to reach 1000
    # If seeds > 1000, we sample. If < 1000, we repeat.
    target_count = TARGET_TOTAL
    
    print(f"Target: {target_count} datasets.")
    
    count = 0
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 10  # Prevent infinite loop
    
    # Use tqdm for progress tracking
    pbar = tqdm(total=target_count, desc="Generating datasets", unit="conv")
    
    while count < target_count:
        # Pick a seed (cycle through)
        seed = seeds[count % len(seeds)]
        
        # Retrieve Context
        context = retrieve_context(seed, chunks, None)
        
        # Prepare Chunk Object for Generator
        chunk_input = {
            "content": context,
            "topic": seed["category"],
            "id": f"seed_{seed['category']}_{count}"
        }
        
        # Generate
        pbar.set_description(f"Generating {seed['category']}")
        result = generator.generate_conversation(chunk_input)
        
        if result:
            consecutive_failures = 0  # Reset on success
            # Flatten structure for final JSON
            item = {
                "instruction": "Multi-turn conversation about UNSIQ", # Metadata
                "input": "",
                "output": json.dumps(result["conversation"], ensure_ascii=False), # Store valid JSON structure
                "text": "", # Will be filled by process_dataset.py formatter later
                "category": seed["category"],
                "source": "synthetic_v1",
                "metadata": result["metadata"]
            }
            generated_data.append(item)
            count += 1
            pbar.update(1)  # Update progress bar
            
            # Save intermediate every 50
            if count % 50 == 0:
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(generated_data, f, ensure_ascii=False, indent=2)
                tqdm.write(f"Saved checkpoint: {count}")
        else:
            consecutive_failures += 1
            tqdm.write(f"Warning: Generation failed ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                tqdm.write("ERROR: Too many consecutive failures. Stopping to prevent infinite loop.")
                break
    
    pbar.close()  # Close progress bar
    
    # Final Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=2)
    
    print("="*60)
    print(f"DONE! Generated {len(generated_data)} conversations.")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
