"""
LLM-as-Judge Evaluation from Inference Results with RAG
Uses existing inference results + RAG context for fact-checking
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm
from datetime import datetime

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    print("Warning: vLLM not installed. Install with: pip install vllm")

# RAG imports
try:
    from src.e5_embedding import E5EmbeddingService
    HAS_RAG = True
except ImportError:
    HAS_RAG = False
    print("Warning: E5 embedding not available. RAG grounding disabled.")

# =============================================================================
# RAG-GROUNDED JUDGE PROMPT
# =============================================================================

JUDGE_SYSTEM_PROMPT = """Anda adalah EVALUATOR KETAT untuk menilai kualitas jawaban AI tentang PMB UNSIQ.

PERINGATAN: Anda HARUS memeriksa FAKTA berdasarkan KONTEKS REFERENSI yang diberikan.
Jika jawaban mengandung informasi yang TIDAK ADA dalam konteks, itu adalah HALUSINASI dan harus diberikan skor RENDAH.

KRITERIA PENILAIAN KETAT (Skala 1-5):

1. **HELPFULNESS** (Kebergunaan):
   1 = Tidak membantu, tidak menjawab pertanyaan
   5 = Menjawab lengkap dan memberikan informasi tambahan berguna

2. **RELEVANCE** (Relevansi):
   1 = Sama sekali tidak relevan
   5 = 100% relevan dengan pertanyaan

3. **FACTUAL_ACCURACY** (Akurasi Faktual) - SANGAT PENTING:
   1 = Banyak kesalahan faktual atau halusinasi
   5 = SEMUA informasi terverifikasi dari konteks referensi

4. **HALLUCINATION_CHECK** (Cek Halusinasi) - KRITIS:
   1 = Banyak informasi yang dikarang/tidak ada di konteks
   5 = ZERO halusinasi, semua dari konteks

5. **COHERENCE** (Koherensi):
   1 = Tidak terstruktur, sulit dipahami
   5 = Sangat jelas dan logis

6. **FLUENCY** (Kefasihan Bahasa):
   1 = Bahasa sangat buruk
   5 = Bahasa profesional dan sopan

Format output HARUS JSON valid!"""

JUDGE_USER_TEMPLATE = """=== KONTEKS REFERENSI (GUNAKAN UNTUK VERIFIKASI FAKTA) ===
{context}

=== PERTANYAAN ===
{question}

=== JAWABAN YANG DINILAI ===
{answer}

INSTRUKSI: Periksa apakah jawaban SESUAI dengan konteks referensi di atas.

Berikan penilaian dalam format JSON:
{{
  "helpfulness": <1-5>,
  "relevance": <1-5>,
  "factual_accuracy": <1-5>,
  "hallucination_check": <1-5>,
  "coherence": <1-5>,
  "fluency": <1-5>,
  "total": <rata-rata dari 6 kriteria>,
  "comment": "<komentar singkat>"
}}"""


# =============================================================================
# RAG CONTEXT RETRIEVAL
# =============================================================================

def load_chunks(chunks_path: str) -> List[Dict]:
    """Load chunks from JSONL file."""
    chunks = []
    if os.path.exists(chunks_path):
        with open(chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        print(f"Loaded {len(chunks)} chunks from {chunks_path}")
    else:
        print(f"Warning: Chunks file not found at {chunks_path}")
    return chunks


def retrieve_context_for_question(question: str, chunks: List[Dict], embed_service, top_k: int = 3) -> str:
    """Retrieve relevant context for a question using semantic search."""
    if not chunks or embed_service is None:
        return ""
    
    try:
        import numpy as np
        query_emb = embed_service.encode_query(question)
        chunk_contents = [c.get("content", "") for c in chunks]
        
        if "embedding" in chunks[0]:
            chunk_embs = np.array([c["embedding"] for c in chunks])
        else:
            chunk_embs = embed_service.encode_passages(chunk_contents)
        
        results = embed_service.find_similar(query_emb, chunk_embs, top_k=top_k, threshold=0.3)
        
        if results:
            selected_chunks = [chunks[idx] for idx, _ in results]
            context = "\n\n---\n\n".join([c.get("content", "") for c in selected_chunks])
            return context
    except Exception as e:
        print(f"Warning: Context retrieval failed: {e}")
    
    return ""


def retrieve_contexts_batch(questions: List[str], chunks: List[Dict], embed_service, top_k: int = 3) -> List[str]:
    """Retrieve contexts for a batch of questions."""
    print("Retrieving RAG contexts...")
    contexts = []
    for q in tqdm(questions, desc="Retrieving contexts"):
        ctx = retrieve_context_for_question(q, chunks, embed_service, top_k)
        contexts.append(ctx)
    return contexts


# =============================================================================
# LLM JUDGE WITH RAG
# =============================================================================

class LLMJudge:
    """vLLM-based LLM-as-Judge with RAG context."""
    
    def __init__(self, model_name: str = "google/gemma-3-12b-it"):
        self.model_name = model_name
        self.llm = None
        
    def load(self):
        """Load the judge model with vLLM."""
        print(f"\nLoading Judge Model: {self.model_name}")
        
        self.llm = LLM(
            model=self.model_name,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            max_num_batched_tokens=16384,
        )
        print("Judge model loaded!")
        
    def _create_prompt(self, question: str, answer: str, context: str = "") -> str:
        """Create RAG-grounded judge prompt."""
        user_prompt = JUDGE_USER_TEMPLATE.format(
            context=context if context else "Tidak ada konteks referensi tersedia.",
            question=question,
            answer=answer
        )
        
        prompt = f"""<bos><start_of_turn>user
{JUDGE_SYSTEM_PROMPT}

{user_prompt}<end_of_turn>
<start_of_turn>model
"""
        return prompt
    
    def _parse_scores(self, text: str) -> Optional[Dict]:
        """Parse JSON scores from judge output."""
        start_idx = text.find('{')
        if start_idx == -1:
            return None
            
        brace_count = 0
        end_idx = -1
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
        
        if end_idx == -1:
            return None
            
        try:
            json_str = text[start_idx:end_idx + 1]
            scores = json.loads(json_str)
            
            required = ["helpfulness", "relevance", "factual_accuracy", "hallucination_check", "coherence", "fluency"]
            if all(k in scores for k in required):
                if "total" not in scores:
                    scores["total"] = sum(scores[k] for k in required) / len(required)
                return scores
        except json.JSONDecodeError:
            pass
            
        return None
    
    def score_batch(self, questions: List[str], answers: List[str], contexts: List[str] = None) -> List[Dict]:
        """Score a batch with RAG contexts."""
        
        if not self.llm:
            self.load()
        
        if contexts is None:
            contexts = [""] * len(questions)
        
        prompts = [self._create_prompt(q, a, c) for q, a, c in zip(questions, answers, contexts)]
        
        sampling_params = SamplingParams(
            max_tokens=512,
            temperature=0.1,
            top_p=0.95,
            stop=["<end_of_turn>"]
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        scores = []
        for output in outputs:
            text = output.outputs[0].text
            parsed = self._parse_scores(text)
            if parsed:
                scores.append(parsed)
            else:
                scores.append({
                    "helpfulness": 0, "relevance": 0, "factual_accuracy": 0,
                    "hallucination_check": 0, "coherence": 0, "fluency": 0, 
                    "total": 0, "comment": "PARSE_ERROR"
                })
        
        return scores


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate_llm_judge(
    input_path: str,
    output_path: str,
    chunks_path: str = "../data/chunks/chunks.jsonl",
    judge_model: str = "google/gemma-3-12b-it",
    batch_size: int = 32,
    rag_top_k: int = 3
):
    """Evaluate inference results with RAG-grounded LLM-as-Judge."""
    
    if not HAS_VLLM:
        print("ERROR: vLLM is required. Install with: pip install vllm")
        return None
    
    print(f"\n{'='*60}")
    print("LLM-AS-JUDGE EVALUATION WITH RAG GROUNDING")
    print(f"{'='*60}")
    
    # Load inference results
    print(f"\nLoading inference results from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    
    prompts = [item["prompt"] for item in data]
    base_responses = [item["base_response"] for item in data]
    finetuned_responses = [item["finetuned_response"] for item in data]
    
    # Load RAG chunks
    chunks = []
    embed_service = None
    
    if HAS_RAG:
        print(f"\nLoading RAG knowledge base from: {chunks_path}")
        chunks = load_chunks(chunks_path)
        if chunks:
            print("Initializing E5 embedding service...")
            embed_service = E5EmbeddingService()
            print(f"RAG enabled: {len(chunks)} chunks loaded")
    else:
        print("Warning: RAG not available, proceeding without fact grounding")
    
    # Retrieve contexts
    contexts = retrieve_contexts_batch(prompts, chunks, embed_service, top_k=rag_top_k)
    print(f"Retrieved contexts for {len(contexts)} questions")
    
    # Initialize judge
    judge = LLMJudge(judge_model)
    judge.load()
    
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    
    # Score BASE model
    print(f"\n{'='*60}")
    print("Scoring BASE MODEL responses with RAG grounding...")
    print(f"{'='*60}")
    
    base_scores = []
    for i in tqdm(range(num_batches), desc="Judging base model"):
        start = i * batch_size
        end = min(start + batch_size, len(prompts))
        batch_scores = judge.score_batch(
            prompts[start:end], 
            base_responses[start:end],
            contexts[start:end]
        )
        base_scores.extend(batch_scores)
    
    # Score FINETUNED model
    print(f"\n{'='*60}")
    print("Scoring FINETUNED MODEL responses with RAG grounding...")
    print(f"{'='*60}")
    
    ft_scores = []
    for i in tqdm(range(num_batches), desc="Judging finetuned model"):
        start = i * batch_size
        end = min(start + batch_size, len(prompts))
        batch_scores = judge.score_batch(
            prompts[start:end], 
            finetuned_responses[start:end],
            contexts[start:end]
        )
        ft_scores.extend(batch_scores)
    
    # Compute statistics
    def compute_stats(scores: List[Dict]) -> Dict:
        dimensions = ["helpfulness", "relevance", "factual_accuracy", "hallucination_check", "coherence", "fluency", "total"]
        stats = {}
        for dim in dimensions:
            values = [s.get(dim, 0) for s in scores if s.get(dim, 0) > 0]
            if values:
                stats[dim] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
            else:
                stats[dim] = {"mean": 0, "min": 0, "max": 0, "count": 0}
        return stats
    
    base_stats = compute_stats(base_scores)
    ft_stats = compute_stats(ft_scores)
    
    # Comparison
    comparison = {}
    for dim in ["helpfulness", "relevance", "factual_accuracy", "hallucination_check", "coherence", "fluency", "total"]:
        base_mean = base_stats.get(dim, {}).get("mean", 0)
        ft_mean = ft_stats.get(dim, {}).get("mean", 0)
        improvement = ft_mean - base_mean
        pct = (improvement / base_mean * 100) if base_mean > 0 else 0
        comparison[dim] = {
            "base": base_mean,
            "finetuned": ft_mean,
            "improvement": improvement,
            "improvement_percent": pct
        }
    
    results = {
        "config": {
            "judge_model": judge_model,
            "chunks_path": chunks_path,
            "num_samples": len(data),
            "rag_enabled": HAS_RAG and len(chunks) > 0,
            "rag_top_k": rag_top_k,
            "timestamp": datetime.now().isoformat(),
        },
        "base_model": {"stats": base_stats},
        "finetuned_model": {"stats": ft_stats},
        "comparison": comparison,
    }
    
    # Save results
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*60)
    print("LLM-AS-JUDGE EVALUATION RESULTS (RAG-GROUNDED)")
    print("="*60)
    print(f"\n{'Dimension':<20} {'Base':<10} {'Finetuned':<12} {'Improvement':<12}")
    print("-"*60)
    for dim in ["helpfulness", "relevance", "factual_accuracy", "hallucination_check", "coherence", "fluency"]:
        b = base_stats[dim]["mean"]
        f = ft_stats[dim]["mean"]
        print(f"{dim:<20} {b:<10.2f} {f:<12.2f} {f-b:+.2f}")
    print("-"*60)
    b_total = base_stats["total"]["mean"]
    f_total = ft_stats["total"]["mean"]
    pct = ((f_total - b_total) / b_total * 100) if b_total > 0 else 0
    print(f"{'TOTAL':<20} {b_total:<10.2f} {f_total:<12.2f} {f_total-b_total:+.2f} ({pct:+.1f}%)")
    print("="*60)
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-Judge with RAG from Inference Results")
    parser.add_argument("--input", type=str, default="./outputs/inference_results_multiturn.json")
    parser.add_argument("--output", type=str, default="./outputs/llm_judge_results.json")
    parser.add_argument("--chunks", type=str, default="../data/chunks/chunks.jsonl")
    parser.add_argument("--judge_model", type=str, default="google/gemma-3-12b-it")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--rag_top_k", type=int, default=3)
    
    args = parser.parse_args()
    
    evaluate_llm_judge(
        args.input, args.output, args.chunks,
        args.judge_model, args.batch_size, args.rag_top_k
    )
