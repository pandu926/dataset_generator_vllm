"""
Batch LLM-as-Judge Evaluation using OpenAI GPT-5 Mini API
- Uses async parallel requests for maximum speed
- Compatible with existing output format
- Supports resume (skip already evaluated models)

Usage:
    python batch_llm_judge_openai.py
"""

import os
import sys
import json
import asyncio
import argparse
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# OpenAI
from openai import AsyncOpenAI

# RAG embedding service
try:
    from src.e5_embedding import E5EmbeddingService
    HAS_RAG = True
except ImportError:
    HAS_RAG = False
    print("Warning: E5 embedding not available. RAG grounding disabled.")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BatchEvalConfig:
    """Batch evaluation configuration."""
    # OpenAI
    openai_api_key: str = ""
    judge_model_name: str = "gpt-5-mini"
    
    # Model paths
    base_model_name: str = "google/gemma-3-1b-it"
    models_dir: str = "./model_hasil_reserach_parameter"
    
    # Paths
    test_dataset_path: str = "../data/final/split/merged_all_categories_test_no_cot.json"
    output_dir: str = "./outputs/batch_evaluation_results"
    chunks_path: str = "../data/chunks/chunks.jsonl"
    
    # RAG Settings
    rag_top_k: int = 3
    use_rag_for_generation: bool = True
    
    # Batch sizes
    generation_batch_size: int = 32
    openai_concurrent_requests: int = 50  # Parallel requests to OpenAI
    max_test_samples: int = 0  # 0 = use ALL samples
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    
    # Resume
    skip_existing: bool = True


# =============================================================================
# JUDGE PROMPTS
# =============================================================================

JUDGE_SYSTEM_PROMPT = """Anda adalah EVALUATOR KETAT untuk menilai kualitas jawaban AI tentang PMB UNSIQ.

PERINGATAN: Anda HARUS memeriksa FAKTA berdasarkan KONTEKS REFERENSI yang diberikan.
Jika jawaban mengandung informasi yang TIDAK ADA dalam konteks, itu adalah HALUSINASI dan harus diberikan skor RENDAH.

KRITERIA PENILAIAN KETAT (Skala 1-5):

1. **HELPFULNESS** (Kebergunaan):
   1 = Tidak membantu, tidak menjawab pertanyaan
   2 = Hanya sedikit relevan
   3 = Menjawab sebagian pertanyaan
   4 = Menjawab dengan baik
   5 = Menjawab lengkap dan memberikan informasi tambahan berguna

2. **RELEVANCE** (Relevansi):
   1 = Sama sekali tidak relevan
   2 = Menyimpang dari topik
   3 = Sebagian relevan
   4 = Mayoritas relevan
   5 = 100% relevan dengan pertanyaan

3. **FACTUAL_ACCURACY** (Akurasi Faktual) - SANGAT PENTING:
   1 = Banyak kesalahan faktual atau halusinasi
   2 = Ada beberapa informasi salah
   3 = Sebagian besar benar tapi ada keraguan
   4 = Hampir semua informasi sesuai konteks
   5 = SEMUA informasi terverifikasi dari konteks referensi

4. **HALLUCINATION_CHECK** (Cek Halusinasi) - KRITIS:
   1 = Banyak informasi yang dikarang/tidak ada di konteks
   2 = Ada beberapa informasi tidak terverifikasi
   3 = Mayoritas dapat diverifikasi
   4 = Hampir tidak ada halusinasi
   5 = ZERO halusinasi, semua dari konteks

5. **COHERENCE** (Koherensi):
   1 = Tidak terstruktur, sulit dipahami
   2 = Struktur lemah
   3 = Cukup jelas
   4 = Terstruktur dengan baik
   5 = Sangat jelas dan logis

6. **FLUENCY** (Kefasihan Bahasa):
   1 = Bahasa sangat buruk
   2 = Ada kesalahan gramatikal
   3 = Bahasa standar
   4 = Bahasa baik dan natural
   5 = Bahasa profesional dan sopan

ATURAN PENILAIAN:
- Jika jawaban mengandung ANGKA/BIAYA yang tidak ada di konteks: FACTUAL_ACCURACY = 1, HALLUCINATION = 1
- Jika jawaban menyebut NAMA INSTITUSI/FAKULTAS yang salah: FACTUAL_ACCURACY = 1
- Jika jawaban mengatakan "tidak tersedia/tidak tahu" padahal info ADA di konteks: HELPFULNESS = 1
- Jika jawaban mengandung info benar dari konteks: berikan skor tinggi

Format output HARUS JSON valid!"""

JUDGE_USER_TEMPLATE = """=== KONTEKS REFERENSI (GUNAKAN UNTUK VERIFIKASI FAKTA) ===
{context}

=== PERTANYAAN ===
{question}

=== JAWABAN YANG DINILAI ===
{answer}

INSTRUKSI: Periksa apakah jawaban SESUAI dengan konteks referensi di atas.
Jika ada informasi dalam jawaban yang TIDAK ADA di konteks, berikan skor rendah untuk accuracy dan hallucination.

Berikan penilaian dalam format JSON:
{{
  "helpfulness": <1-5>,
  "relevance": <1-5>,
  "factual_accuracy": <1-5>,
  "hallucination_check": <1-5>,
  "coherence": <1-5>,
  "fluency": <1-5>,
  "total": <rata-rata dari 6 kriteria>,
  "detected_hallucinations": "<list info yang tidak ada di konteks atau 'none'>",
  "verified_facts": "<list info yang terverifikasi dari konteks>",
  "comment": "<komentar singkat>"
}}"""

RAG_GENERATION_TEMPLATE = """Berikut adalah konteks referensi yang relevan untuk menjawab pertanyaan:

=== KONTEKS REFERENSI ===
{context}
=== AKHIR KONTEKS ===

Berdasarkan konteks di atas, jawab pertanyaan berikut dengan akurat dan sopan.
Jika informasi tidak ada di konteks, katakan bahwa Anda tidak memiliki informasi tersebut.

Pertanyaan: {question}"""


# =============================================================================
# OPENAI ASYNC JUDGE
# =============================================================================

class OpenAIAsyncJudge:
    """Async OpenAI-based LLM-as-Judge for high-speed parallel scoring."""
    
    def __init__(self, api_key: str, model_name: str, config: BatchEvalConfig):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model_name = model_name
        self.config = config
        self.semaphore = asyncio.Semaphore(config.openai_concurrent_requests)
    
    def _create_messages(self, question: str, answer: str, context: str = "") -> List[Dict]:
        """Create messages for judging."""
        user_prompt = JUDGE_USER_TEMPLATE.format(
            context=context if context else "Tidak ada konteks referensi tersedia.",
            question=question,
            answer=answer
        )
        return [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    
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
    
    async def _score_single(self, question: str, answer: str, context: str) -> Dict:
        """Score a single Q&A pair."""
        async with self.semaphore:
            try:
                messages = self._create_messages(question, answer, context)
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_completion_tokens=2048,  # GPT-5 Mini needs ~128-256 for reasoning + output
                )
                text = response.choices[0].message.content
                parsed = self._parse_scores(text)
                if parsed:
                    return parsed
                else:
                    return {
                        "helpfulness": 0, "relevance": 0, "factual_accuracy": 0,
                        "hallucination_check": 0, "coherence": 0, "fluency": 0,
                        "total": 0, "comment": "PARSE_ERROR", "raw_output": text[:300]
                    }
            except Exception as e:
                return {
                    "helpfulness": 0, "relevance": 0, "factual_accuracy": 0,
                    "hallucination_check": 0, "coherence": 0, "fluency": 0,
                    "total": 0, "comment": f"API_ERROR: {str(e)}"
                }
    
    async def score_batch_async(self, questions: List[str], answers: List[str], contexts: List[str]) -> List[Dict]:
        """Score all Q&A pairs in parallel."""
        tasks = [
            self._score_single(q, a, c)
            for q, a, c in zip(questions, answers, contexts)
        ]
        results = await tqdm_asyncio.gather(*tasks, desc="Scoring with OpenAI")
        return results


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
        query_emb = embed_service.encode_query(question)
        chunk_contents = [c.get("content", "") for c in chunks]
        
        if "embedding" in chunks[0]:
            import numpy as np
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
    contexts = []
    for q in tqdm(questions, desc="Retrieving RAG contexts"):
        ctx = retrieve_context_for_question(q, chunks, embed_service, top_k)
        contexts.append(ctx)
    return contexts


# =============================================================================
# MODEL RESPONSE GENERATORS
# =============================================================================

def load_finetuned_model(base_model_name: str, adapter_path: str, dtype: torch.dtype = torch.bfloat16):
    """Load finetuned model with LoRA adapters."""
    print(f"\nLoading finetuned model from: {adapter_path}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    
    model = PeftModel.from_pretrained(model, adapter_path)
    
    print("Finetuned model loaded!")
    return model, tokenizer


def generate_responses_batch_with_rag(
    model, tokenizer, prompts: List[str], contexts: List[str],
    max_new_tokens: int = 2048,
) -> List[str]:
    """Generate responses for a batch of prompts WITH RAG context."""
    
    formatted_prompts = []
    for prompt, context in zip(prompts, contexts):
        if context:
            augmented_prompt = RAG_GENERATION_TEMPLATE.format(context=context, question=prompt)
        else:
            augmented_prompt = prompt
        formatted_prompts.append(f"<bos><start_of_turn>user\n{augmented_prompt}<end_of_turn>\n<start_of_turn>model\n")
    
    inputs = tokenizer(
        formatted_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048,
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic generation for consistent evaluation
            pad_token_id=tokenizer.pad_token_id,
        )
    
    responses = []
    for output in outputs:
        full_response = tokenizer.decode(output, skip_special_tokens=False)
        if "<start_of_turn>model\n" in full_response:
            response = full_response.split("<start_of_turn>model\n")[-1]
            if "<end_of_turn>" in response:
                response = response.split("<end_of_turn>")[0]
            responses.append(response.strip())
        else:
            responses.append("")
    
    return responses


def generate_all_responses_with_rag(
    model, tokenizer, test_samples: List[Dict], contexts: List[str],
    batch_size: int = 32, **gen_kwargs
) -> List[str]:
    """Generate responses for all test samples using batch processing WITH RAG."""
    
    prompts = [s["prompt"] for s in test_samples]
    all_responses = []
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Generating responses"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]
        batch_contexts = contexts[start_idx:end_idx]
        batch_responses = generate_responses_batch_with_rag(model, tokenizer, batch_prompts, batch_contexts, **gen_kwargs)
        all_responses.extend(batch_responses)
    
    return all_responses


# =============================================================================
# DATA LOADING
# =============================================================================

def load_test_data(dataset_path: str, max_samples: int = 0) -> List[Dict]:
    """Load test data and extract FIRST TURN ONLY from each conversation."""
    print(f"\nLoading test data from: {dataset_path}")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if max_samples and max_samples > 0 and len(data) > max_samples:
        test_data = data[:max_samples]
    else:
        test_data = data
    
    test_samples = []
    for item in test_data:
        conversation = item.get("conversation", [])
        if len(conversation) >= 2:
            first_user_msg = None
            first_model_response = None
            
            for turn in conversation:
                if turn["role"] == "user" and first_user_msg is None:
                    content = turn.get("content", "").strip()
                    if content:
                        first_user_msg = content
                elif turn["role"] == "model" and first_user_msg is not None and first_model_response is None:
                    content = turn.get("content", "").strip()
                    if content:
                        first_model_response = content
                        break
            
            if first_user_msg and first_model_response:
                test_samples.append({
                    "id": item.get("id", ""),
                    "prompt": first_user_msg,
                    "expected": first_model_response,
                    "category": item.get("category", ""),
                })
    
    print(f"Loaded {len(test_samples)} test samples (first turn only)")
    return test_samples


# =============================================================================
# DISCOVER MODELS
# =============================================================================

def discover_finetuned_models(models_dir: str) -> List[Dict]:
    """Discover all fine-tuned models in the given directory."""
    models = []
    
    if not os.path.exists(models_dir):
        print(f"ERROR: Models directory not found: {models_dir}")
        return models
    
    for folder in sorted(os.listdir(models_dir)):
        folder_path = os.path.join(models_dir, folder)
        final_model_path = os.path.join(folder_path, "final_model")
        experiment_info_path = os.path.join(folder_path, "experiment_info.json")
        
        if os.path.isdir(final_model_path):
            model_info = {
                "name": folder,
                "adapter_path": final_model_path,
                "experiment_info": None,
            }
            
            if os.path.exists(experiment_info_path):
                with open(experiment_info_path, "r", encoding="utf-8") as f:
                    model_info["experiment_info"] = json.load(f)
            
            models.append(model_info)
            print(f"  Found model: {folder}")
    
    print(f"\nTotal models found: {len(models)}")
    return models


# =============================================================================
# EVALUATE SINGLE MODEL
# =============================================================================

async def evaluate_single_model(
    model_info: Dict,
    test_samples: List[Dict],
    contexts: List[str],
    judge: OpenAIAsyncJudge,
    config: BatchEvalConfig,
) -> Dict:
    """Evaluate a single fine-tuned model."""
    
    model_name = model_info["name"]
    adapter_path = model_info["adapter_path"]
    
    print(f"\n{'='*70}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*70}")
    
    prompts = [s["prompt"] for s in test_samples]
    
    gen_kwargs = {
        "max_new_tokens": config.max_new_tokens,
    }
    
    # Load and generate from fine-tuned model
    ft_model, ft_tokenizer = load_finetuned_model(config.base_model_name, adapter_path)
    
    responses = generate_all_responses_with_rag(
        ft_model, ft_tokenizer, test_samples, contexts,
        batch_size=config.generation_batch_size, **gen_kwargs
    )
    
    # Free model memory
    del ft_model, ft_tokenizer
    torch.cuda.empty_cache()
    
    # Score with OpenAI LLM-as-Judge (async parallel)
    print(f"\nScoring {model_name} responses with OpenAI {config.judge_model_name}...")
    scores = await judge.score_batch_async(prompts, responses, contexts)
    
    # Compute statistics
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
    
    # Build result (matching existing format)
    result = {
        "model_name": model_name,
        "adapter_path": adapter_path,
        "experiment_info": model_info.get("experiment_info"),
        "config": {
            "base_model": config.base_model_name,
            "finetuned_model": adapter_path,
            "judge_model": config.judge_model_name,
            "num_samples": len(test_samples),
            "rag_enabled": config.use_rag_for_generation,
            "rag_top_k": config.rag_top_k,
            "timestamp": datetime.now().isoformat(),
        },
        "finetuned_model": {
            "stats": stats,
            "sample_responses": responses[:5],
            "sample_scores": scores[:5],
        },
        "detailed_results": [],
    }
    
    # Build detailed results for each sample
    for sample, resp, sc, ctx in zip(test_samples, responses, scores, contexts):
        result["detailed_results"].append({
            "id": sample["id"],
            "category": sample.get("category", ""),
            "prompt": sample["prompt"],
            "expected_response": sample.get("expected", ""),
            "rag_context_used": ctx[:500] + "..." if len(ctx) > 500 else ctx,
            "finetuned_model_response": resp,
            "finetuned_model_scores": sc,
        })
    
    # Save per-model result file
    model_output_path = os.path.join(config.output_dir, f"llm_judge_{model_name}.json")
    os.makedirs(config.output_dir, exist_ok=True)
    with open(model_output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved result to: {model_output_path}")
    
    # Print summary
    print(f"\n{'-'*50}")
    print(f"Results for {model_name}:")
    print(f"{'-'*50}")
    for dim in ["helpfulness", "relevance", "factual_accuracy", "hallucination_check", "coherence", "fluency"]:
        mean = stats.get(dim, {}).get("mean", 0)
        print(f"  {dim.replace('_', ' ').title()}: {mean:.2f}")
    print(f"  TOTAL: {stats.get('total', {}).get('mean', 0):.2f}")
    
    return result


# =============================================================================
# MAIN BATCH EVALUATION
# =============================================================================

async def batch_evaluate_async(config: BatchEvalConfig):
    """Main batch evaluation function for all fine-tuned models."""
    
    print("\n" + "="*70)
    print("BATCH LLM-AS-JUDGE EVALUATION (OpenAI GPT-5 Mini)")
    print("="*70)
    print(f"Judge Model: {config.judge_model_name}")
    print(f"Base Model: {config.base_model_name}")
    print(f"Models Directory: {config.models_dir}")
    print(f"Concurrent Requests: {config.openai_concurrent_requests}")
    print(f"RAG Enabled: {config.use_rag_for_generation}")
    print("="*70)
    
    # Discover all fine-tuned models
    print("\nDiscovering fine-tuned models...")
    models = discover_finetuned_models(config.models_dir)
    
    if not models:
        print("ERROR: No fine-tuned models found!")
        return None
    
    # Check for existing results (resume support)
    if config.skip_existing:
        existing_results = []
        for f in os.listdir(config.output_dir) if os.path.exists(config.output_dir) else []:
            if f.startswith("llm_judge_") and f.endswith(".json"):
                model_name = f.replace("llm_judge_", "").replace(".json", "")
                existing_results.append(model_name)
        
        models_to_eval = [m for m in models if m["name"] not in existing_results]
        print(f"\nSkipping {len(existing_results)} already evaluated models")
        print(f"Models to evaluate: {len(models_to_eval)}")
        models = models_to_eval
        
        if not models:
            print("All models already evaluated!")
            return None
    
    # Load test data
    test_samples = load_test_data(config.test_dataset_path, config.max_test_samples)
    prompts = [s["prompt"] for s in test_samples]
    
    # Load RAG system
    print("\n" + "-"*40)
    print("Loading RAG Knowledge Base")
    print("-"*40)
    
    chunks = []
    embed_service = None
    contexts = [""] * len(prompts)
    
    if HAS_RAG and config.use_rag_for_generation:
        chunks = load_chunks(config.chunks_path)
        if chunks:
            print("Initializing E5 embedding service...")
            embed_service = E5EmbeddingService()
            print(f"RAG enabled: {len(chunks)} chunks loaded")
            contexts = retrieve_contexts_batch(prompts, chunks, embed_service, top_k=config.rag_top_k)
            print(f"Retrieved contexts for {len(contexts)} questions")
    
    # Initialize judge
    judge = OpenAIAsyncJudge(config.openai_api_key, config.judge_model_name, config)
    
    # Evaluate each model
    all_results = []
    
    for i, model_info in enumerate(models):
        print(f"\n\n{'#'*70}")
        print(f"MODEL {i+1}/{len(models)}: {model_info['name']}")
        print(f"{'#'*70}")
        
        result = await evaluate_single_model(
            model_info=model_info,
            test_samples=test_samples,
            contexts=contexts,
            judge=judge,
            config=config,
        )
        all_results.append(result)
    
    # Build summary
    summary = {
        "config": {
            "base_model": config.base_model_name,
            "judge_model": config.judge_model_name,
            "num_models_evaluated": len(models),
            "num_test_samples": len(test_samples),
            "rag_enabled": config.use_rag_for_generation,
            "rag_top_k": config.rag_top_k,
            "timestamp": datetime.now().isoformat(),
        },
        "model_results": [{"name": r["model_name"], "stats": r["finetuned_model"]["stats"]} for r in all_results],
        "ranking": [],
    }
    
    # Create ranking
    ranking = sorted(
        [(r["model_name"], r["finetuned_model"]["stats"]["total"]["mean"]) for r in all_results],
        key=lambda x: x[1],
        reverse=True
    )
    summary["ranking"] = [{"rank": i+1, "model": m, "score": s} for i, (m, s) in enumerate(ranking)]
    
    # Save summary
    os.makedirs(config.output_dir, exist_ok=True)
    output_path = os.path.join(config.output_dir, f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Print final summary
    print("\n\n" + "="*70)
    print("FINAL RANKING - ALL FINE-TUNED MODELS")
    print("="*70)
    print(f"\n{'Rank':<6} {'Model Name':<40} {'Total Score':<12}")
    print("-"*70)
    for item in summary["ranking"]:
        print(f"{item['rank']:<6} {item['model']:<40} {item['score']:.3f}")
    print("="*70)
    print(f"\nSummary saved to: {output_path}")
    
    return summary


def batch_evaluate(config: BatchEvalConfig):
    """Wrapper to run async batch evaluation."""
    return asyncio.run(batch_evaluate_async(config))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch LLM-as-Judge Evaluation with OpenAI GPT-5 Mini")
    parser.add_argument("--max_samples", type=int, default=0, help="Max test samples (0=all)")
    parser.add_argument("--concurrent", type=int, default=50, help="Concurrent OpenAI requests")
    parser.add_argument("--gen_batch", type=int, default=32, help="Generation batch size")
    parser.add_argument("--rag_top_k", type=int, default=3, help="RAG top-k chunks")
    parser.add_argument("--no_skip", action="store_true", help="Don't skip existing results")
    
    args = parser.parse_args()
    
    config = BatchEvalConfig(
        max_test_samples=args.max_samples,
        openai_concurrent_requests=args.concurrent,
        generation_batch_size=args.gen_batch,
        rag_top_k=args.rag_top_k,
        skip_existing=not args.no_skip,
    )
    
    batch_evaluate(config)
