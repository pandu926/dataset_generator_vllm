import os
import json
import asyncio
import argparse
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
from collections import defaultdict
from openai import AsyncOpenAI
from tqdm import tqdm

# Try importing sentence_transformers for RAG
try:
    from sentence_transformers import SentenceTransformer
    HAS_RAG_LIB = True
except ImportError:
    HAS_RAG_LIB = False

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_RESULTS_FILE = "llm_judge_results.json"
DEFAULT_OUTPUT_FILE = "llm_judge_results_openai.json"
DEFAULT_CHUNKS_FILE = "../data/chunks/chunks.jsonl"
DEFAULT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# =============================================================================
# RAG SERVICE
# =============================================================================

class RAGService:
    def __init__(self, chunks_path: str, model_name: str = EMBEDDING_MODEL):
        self.chunks = []
        self.embeddings = None
        self.model = None
        
        if not HAS_RAG_LIB:
            print("[RAG] sentence-transformers not installed. RAG disabled.")
            return

        if not os.path.exists(chunks_path):
            print(f"[RAG] Chunks file not found at {chunks_path}. RAG disabled.")
            return

        print(f"[RAG] Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        print(f"[RAG] Loading chunks from {chunks_path}...")
        self._load_chunks(chunks_path)
        
        print(f"[RAG] Precomputing embeddings for {len(self.chunks)} chunks...")
        texts = [f"passage: {c['content']}" for c in self.chunks]
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)
        print("[RAG] Initialization complete.")

    def _load_chunks(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.chunks.append(json.loads(line))

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if self.model is None or self.embeddings is None:
            return []
        
        query_text = f"query: {query}"
        query_emb = self.model.encode(query_text, normalize_embeddings=True)
        
        # Cosine similarity
        scores = np.dot(self.embeddings, query_emb)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [self.chunks[i]['content'] for i in top_indices]

# =============================================================================
# OPENAI JUDGE
# =============================================================================

SYSTEM_PROMPT = """You are an impartial and expert judge evaluating the quality of AI assistant responses for a university chatbot (UNSIQ).
You will be given a user question, reference context (RAG), an expected response, and an AI response to evaluate.
ALL YOUR COMMENTS MUST BE IN INDONESIAN.

Your task is to score the AI response on a scale of 1-5 for the following criteria:
1. Helpfulness: Does it answer the user's intent?
2. Relevance: Is it relevant to the question and context?
3. Factual Accuracy: Is the information factually correct based on the context?
4. Hallucination Check: Does it invent information not present in the context? (5 = No hallucination, 1 = Severe hallucination)
5. Coherence: Is the response logically structured?
6. Fluency: Is the language natural and grammatically correct?

You must also provide:
- detected_hallucinations: List any specific hallucinatory claims (or "none").
- verified_facts: List facts that are correctly verified against the context.
- comment: Brief explanation of the scores (MUST BE IN INDONESIAN).

Return your evaluation in valid JSON format ONLY.
Format:
{
  "helpfulness": <int>,
  "relevance": <int>,
  "factual_accuracy": <int>,
  "hallucination_check": <int>,
  "coherence": <int>,
  "fluency": <int>,
  "total": <float average>,
  "detected_hallucinations": <string or list>,
  "verified_facts": <string or list>,
  "comment": <string>
}
"""

async def evaluate_single_response(
    client: AsyncOpenAI, 
    model: str, 
    prompt: str, 
    response: str, 
    expected: str, 
    context: List[str]
) -> Dict[str, Any]:
    
    rag_text = "\n\n".join(context) if context else "No distinct context provided."
    
    user_content = f"""
[QUESTION]
{prompt}

[CONTEXT / KNOWLEDGE BASE]
{rag_text}

[EXPECTED RESPONSE (Reference only)]
{expected}

[AI RESPONSE TO EVALUATE]
{response}

Evaluate the AI RESPONSE.
"""

    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        content = completion.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"Error evaluating: {e}")
        return {
            "helpfulness": 0, "relevance": 0, "factual_accuracy": 0, 
            "hallucination_check": 0, "coherence": 0, "fluency": 0, "total": 0,
            "comment": f"Evaluation failed: {str(e)}"
        }

# =============================================================================
# MAIN LOGIC
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Evaluate existing results with OpenAI GPT-4o-mini")
    parser.add_argument("--input", default=DEFAULT_RESULTS_FILE, help="Input results JSON file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Output JSON file")
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS_FILE, help="Path to RAG chunks")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model to use")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API Key")
    parser.add_argument("--batch_size", type=int, default=20, help="Concurrent requests")
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("Error: OpenAI API Key is required (set OPENAI_API_KEY env var or pass --api_key)")
        return

    # Load Data
    print(f"Loading results from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    detailed_results = data.get("detailed_results", [])
    if not detailed_results:
        print("No detailed results found in input file.")
        return
        
    print(f"Found {len(detailed_results)} samples to evaluate.")

    # Init RAG
    rag = RAGService(args.chunks)
    
    # Init OpenAI
    client = AsyncOpenAI(api_key=args.api_key)
    
    # Process
    sem = asyncio.Semaphore(args.batch_size)
    
    async def process_item(item):
        async with sem:
            prompt = item['prompt']
            expected = item.get('expected_response', '')
            base_resp = item['base_model_response']
            ft_resp = item['finetuned_model_response']
            
            # 1. Retrieve Context
            context = rag.retrieve(prompt, top_k=4)
            
            # 2. Evaluate Base & Finetuned in parallel
            task_base = evaluate_single_response(client, args.model, prompt, base_resp, expected, context)
            task_ft = evaluate_single_response(client, args.model, prompt, ft_resp, expected, context)
            
            score_base, score_ft = await asyncio.gather(task_base, task_ft)
            
            # Update item
            item['base_model_scores'] = score_base
            item['finetuned_model_scores'] = score_ft
            
            # Recalculate comparison
            base_total = score_base.get('total', 0)
            ft_total = score_ft.get('total', 0)
            
            item['score_comparison'] = {
                "base_total": base_total,
                "finetuned_total": ft_total,
                "winner": "finetuned" if ft_total > base_total else ("base" if base_total > ft_total else "tie")
            }
            
            return item

    print("Starting evaluation...")
    tasks = [process_item(item) for item in detailed_results]
    
    # Run with progress bar
    new_results = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating"):
        result = await f
        new_results.append(result)
        
    # Re-sort to match original order (optional, but good for ID match)
    # Assuming IDs are unique
    id_map = {item['id']: item for item in new_results}
    final_detailed_results = [id_map[item['id']] for item in detailed_results if item['id'] in id_map]
    
    # Recalculate Global Stats
    def calculate_stats(scores_list):
        dimensions = ["helpfulness", "relevance", "factual_accuracy", "hallucination_check", "coherence", "fluency", "total"]
        stats = {}
        for dim in dimensions:
            vals = [s.get(dim, 0) for s in scores_list]
            if vals:
                stats[dim] = {
                    "mean": sum(vals) / len(vals),
                    "min": min(vals),
                    "max": max(vals),
                    "count": len(vals)
                }
        return stats

    base_scores_all = [x['base_model_scores'] for x in final_detailed_results]
    ft_scores_all = [x['finetuned_model_scores'] for x in final_detailed_results]
    
    data['base_model']['stats'] = calculate_stats(base_scores_all)
    data['finetuned_model']['stats'] = calculate_stats(ft_scores_all)
    data['detailed_results'] = final_detailed_results
    
    # Update Comparison
    comp = {}
    for dim in ["helpfulness", "relevance", "factual_accuracy", "hallucination_check", "coherence", "fluency", "total"]:
        base_mean = data['base_model']['stats'][dim]['mean']
        ft_mean = data['finetuned_model']['stats'][dim]['mean']
        imp = ft_mean - base_mean
        pct = (imp / base_mean * 100) if base_mean > 0 else 0
        comp[dim] = {
            "base": base_mean,
            "finetuned": ft_mean,
            "improvement": imp,
            "improvement_percent": pct
        }
    data['comparison'] = comp
    
    # Save
    print(f"Saving results to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
