"""
LLM-as-Judge Evaluation Script
Compare base model vs fine-tuned model using Gemma-3-12B as judge
5-Point Likert Scale (Research Best Practices: G-Eval, MT-Bench)
Optimized for A100 80GB with vLLM batch processing

References:
- G-Eval: NLG Evaluation using GPT-4 (Liu et al., 2023)
- MT-Bench: Multi-turn Benchmark (Zheng et al., 2023)
- Judging LLM-as-Judge (Zheng et al., 2023)
"""

import os
import sys
import json
import torch
import argparse
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    print("Warning: vLLM not installed. Install with: pip install vllm")

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# RAG imports
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
class EvalConfig:
    """Evaluation configuration."""
    # Models
    base_model_name: str = "google/gemma-3-1b-it"
    finetuned_model_path: str = "./outputs/gemma3-1b-qlora-sft/final_model"
    judge_model_name: str = "google/gemma-3-12b-it"
    
    # Dataset
    test_dataset_path: str = "../data/merged/multiturn_test.json"
    output_path: str = "./outputs/llm_judge_evaluation_results.json"
    
    # RAG Configuration
    chunks_path: str = "../data/chunks/chunks.jsonl"  # For fact-grounded evaluation
    rag_top_k: int = 3  # Number of chunks to retrieve per question
    
    # Batch processing - A100 80GB optimized
    generation_batch_size: int = 16   # For generating responses
    judge_batch_size: int = 32        # For judging (smaller input/output)
    max_test_samples: int = 100       # Number of samples to evaluate
    
    # Generation parameters
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    
    # vLLM Settings for A100 80GB
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    max_num_batched_tokens: int = 16384


# =============================================================================
# STRICT RAG-GROUNDED JUDGE PROMPT (Fact-Checking Mode)
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


# =============================================================================
# VLLM JUDGE ENGINE
# =============================================================================

class VLLMJudge:
    """vLLM-based LLM-as-Judge for efficient batch scoring."""
    
    def __init__(self, model_name: str, config: EvalConfig):
        self.model_name = model_name
        self.config = config
        self.llm = None
        
    def load(self):
        """Load the judge model with vLLM."""
        print(f"\n{'='*60}")
        print(f"Loading Judge Model: {self.model_name}")
        print(f"{'='*60}")
        
        self.llm = LLM(
            model=self.model_name,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
        )
        print("Judge model loaded!")
        
    def _create_judge_prompt(self, question: str, answer: str, context: str = "") -> str:
        """Create prompt for judging with RAG context."""
        user_prompt = JUDGE_USER_TEMPLATE.format(
            context=context if context else "Tidak ada konteks referensi tersedia.",
            question=question,
            answer=answer
        )
        
        # Format for Gemma 3 chat
        prompt = f"""<bos><start_of_turn>user
{JUDGE_SYSTEM_PROMPT}

{user_prompt}<end_of_turn>
<start_of_turn>model
"""
        return prompt
    
    def _parse_scores(self, text: str) -> Optional[Dict]:
        """Parse JSON scores from judge output."""
        import re
        
        # Find JSON in output
        start_idx = text.find('{')
        if start_idx == -1:
            return None
            
        # Find matching closing brace
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
            
            # Validate required fields - updated for 6 criteria
            required = ["helpfulness", "relevance", "factual_accuracy", "hallucination_check", "coherence", "fluency"]
            # Also accept old format for backward compatibility
            old_required = ["helpfulness", "relevance", "accuracy", "coherence", "fluency"]
            
            if all(k in scores for k in required):
                # Calculate total if not present (6 criteria)
                if "total" not in scores:
                    scores["total"] = sum(scores[k] for k in required) / len(required)
                return scores
            elif all(k in scores for k in old_required):
                # Backward compatibility - convert old format
                scores["factual_accuracy"] = scores.get("accuracy", 0)
                scores["hallucination_check"] = scores.get("accuracy", 0)  # Use accuracy as proxy
                if "total" not in scores:
                    scores["total"] = sum(scores.get(k, 0) for k in old_required) / len(old_required)
                return scores
        except json.JSONDecodeError:
            pass
            
        return None
    
    def score_batch(self, questions: List[str], answers: List[str], contexts: List[str] = None) -> List[Dict]:
        """Score a batch of question-answer pairs with optional RAG contexts."""
        
        if not self.llm:
            self.load()
        
        # Default contexts if not provided
        if contexts is None:
            contexts = [""] * len(questions)
        
        # Create prompts with context
        prompts = [
            self._create_judge_prompt(q, a, c) 
            for q, a, c in zip(questions, answers, contexts)
        ]
        
        # Sampling params for judging (low temperature for consistency)
        sampling_params = SamplingParams(
            max_tokens=512,  # Increased for detailed hallucination analysis
            temperature=0.1,
            top_p=0.95,
            stop=["<end_of_turn>"]
        )
        
        # Generate scores
        outputs = self.llm.generate(prompts, sampling_params)
        
        # Parse results
        scores = []
        for output in outputs:
            text = output.outputs[0].text
            parsed = self._parse_scores(text)
            if parsed:
                scores.append(parsed)
            else:
                # Default scores for failed parsing
                scores.append({
                    "helpfulness": 0,
                    "relevance": 0,
                    "factual_accuracy": 0,
                    "hallucination_check": 0,
                    "coherence": 0,
                    "fluency": 0,
                    "total": 0,
                    "comment": "PARSE_ERROR",
                    "raw_output": text[:300]
                })
        
        return scores


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


def retrieve_context_for_question(
    question: str, 
    chunks: List[Dict], 
    embed_service, 
    top_k: int = 3
) -> str:
    """Retrieve relevant context for a question using semantic search."""
    if not chunks or embed_service is None:
        return ""
    
    try:
        # Encode query
        query_emb = embed_service.encode_query(question)
        
        # Get chunk contents and their embeddings
        chunk_contents = [c.get("content", "") for c in chunks]
        
        # Check if chunks have pre-computed embeddings
        if "embedding" in chunks[0]:
            import numpy as np
            chunk_embs = np.array([c["embedding"] for c in chunks])
        else:
            # Compute embeddings on the fly
            chunk_embs = embed_service.encode_passages(chunk_contents)
        
        # Find top-k similar chunks
        results = embed_service.find_similar(query_emb, chunk_embs, top_k=top_k, threshold=0.3)
        
        if results:
            selected_chunks = [chunks[idx] for idx, _ in results]
            context = "\n\n---\n\n".join([c.get("content", "") for c in selected_chunks])
            return context
    except Exception as e:
        print(f"Warning: Context retrieval failed: {e}")
    
    return ""


def retrieve_contexts_batch(
    questions: List[str],
    chunks: List[Dict],
    embed_service,
    top_k: int = 3
) -> List[str]:
    """Retrieve contexts for a batch of questions."""
    contexts = []
    for q in questions:
        ctx = retrieve_context_for_question(q, chunks, embed_service, top_k)
        contexts.append(ctx)
    return contexts


# =============================================================================
# MODEL RESPONSE GENERATORS
# =============================================================================

def load_base_model(model_name: str, dtype: torch.dtype = torch.bfloat16):
    """Load base model with 4-bit quantization."""
    print(f"\nLoading base model: {model_name}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    
    print("Base model loaded!")
    return model, tokenizer


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


def generate_responses_batch(
    model, 
    tokenizer, 
    prompts: List[str],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> List[str]:
    """Generate responses for a batch of prompts."""
    
    # Format prompts for Gemma
    formatted_prompts = [
        f"<bos><start_of_turn>user\n{p}<end_of_turn>\n<start_of_turn>model\n"
        for p in prompts
    ]
    
    # Tokenize batch
    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode responses
    responses = []
    for output in outputs:
        full_response = tokenizer.decode(output, skip_special_tokens=False)
        
        # Extract model response
        if "<start_of_turn>model\n" in full_response:
            response = full_response.split("<start_of_turn>model\n")[-1]
            if "<end_of_turn>" in response:
                response = response.split("<end_of_turn>")[0]
            responses.append(response.strip())
        else:
            responses.append("")
    
    return responses


def generate_all_responses(
    model, 
    tokenizer, 
    test_samples: List[Dict],
    batch_size: int = 16,
    **gen_kwargs
) -> List[str]:
    """Generate responses for all test samples using batch processing."""
    
    prompts = [s["prompt"] for s in test_samples]
    all_responses = []
    
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Generating responses"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]
        
        batch_responses = generate_responses_batch(
            model, tokenizer, batch_prompts, **gen_kwargs
        )
        all_responses.extend(batch_responses)
    
    return all_responses


# =============================================================================
# DATA LOADING
# =============================================================================

def load_test_data(dataset_path: str, max_samples: int = 0) -> List[Dict]:
    """Load test data and extract prompts with expected responses.
    Args:
        max_samples: 0 or None = use ALL samples
    """
    print(f"\nLoading test data from: {dataset_path}")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # max_samples=0 means use ALL samples
    if max_samples and max_samples > 0 and len(data) > max_samples:
        test_data = data[-max_samples:]
    else:
        test_data = data  # Use all
    
    # Extract first user message from each conversation
    test_samples = []
    for item in test_data:
        conversation = item.get("conversation", [])
        if len(conversation) >= 2:
            user_msg = None
            expected_response = None
            
            for turn in conversation:
                if turn["role"] == "user" and user_msg is None:
                    user_msg = turn["content"]
                elif turn["role"] == "model" and user_msg is not None and expected_response is None:
                    expected_response = turn["content"]
                    break
            
            if user_msg and expected_response:
                test_samples.append({
                    "id": item.get("id", ""),
                    "prompt": user_msg,
                    "expected": expected_response,
                    "category": item.get("category", ""),
                })
    
    print(f"Loaded {len(test_samples)} test samples")
    return test_samples


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate(config: EvalConfig = None):
    """Main evaluation function comparing base vs fine-tuned model."""
    
    if config is None:
        config = EvalConfig()
    
    if not HAS_VLLM:
        print("ERROR: vLLM is required for LLM-as-Judge evaluation")
        print("Install with: pip install vllm")
        return None
    
    print("\n" + "="*60)
    print("LLM-AS-JUDGE EVALUATION: BASE vs FINE-TUNED MODEL")
    print("5-Point Likert Scale (Helpfulness, Relevance, Accuracy, Coherence, Fluency)")
    print("="*60)
    print(f"Judge Model: {config.judge_model_name}")
    print(f"Base Model: {config.base_model_name}")
    print(f"Finetuned Model: {config.finetuned_model_path}")
    print(f"Test Samples: {config.max_test_samples}")
    print("="*60)
    
    # Load test data
    test_samples = load_test_data(config.test_dataset_path, config.max_test_samples)
    prompts = [s["prompt"] for s in test_samples]
    
    results = {
        "config": {
            "base_model": config.base_model_name,
            "finetuned_model": config.finetuned_model_path,
            "judge_model": config.judge_model_name,
            "num_samples": len(test_samples),
            "timestamp": datetime.now().isoformat(),
        },
        "base_model": {},
        "finetuned_model": {},
        "comparison": {},
        "detailed_results": [],
    }
    
    gen_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
    }
    
    # ===========================================
    # Generate from BASE MODEL
    # ===========================================
    print("\n" + "-"*40)
    print("STEP 1: Generating responses from BASE MODEL")
    print("-"*40)
    
    base_model, base_tokenizer = load_base_model(config.base_model_name)
    
    base_responses = generate_all_responses(
        base_model, base_tokenizer, test_samples,
        batch_size=config.generation_batch_size, **gen_kwargs
    )
    
    # Free memory
    del base_model
    torch.cuda.empty_cache()
    
    # ===========================================
    # Generate from FINETUNED MODEL
    # ===========================================
    print("\n" + "-"*40)
    print("STEP 2: Generating responses from FINETUNED MODEL")
    print("-"*40)
    
    ft_model, ft_tokenizer = load_finetuned_model(
        config.base_model_name, config.finetuned_model_path
    )
    
    ft_responses = generate_all_responses(
        ft_model, ft_tokenizer, test_samples,
        batch_size=config.generation_batch_size, **gen_kwargs
    )
    
    # Free memory
    del ft_model
    torch.cuda.empty_cache()
    
    # ===========================================
    # Score with LLM-as-JUDGE (RAG-GROUNDED)
    # ===========================================
    print("\n" + "-"*40)
    print("STEP 3: Scoring with RAG-Grounded LLM-as-Judge")
    print("-"*40)
    
    # Load RAG chunks for fact-grounded evaluation
    chunks = []
    embed_service = None
    
    if HAS_RAG:
        print(f"\nLoading RAG knowledge base from: {config.chunks_path}")
        chunks = load_chunks(config.chunks_path)
        if chunks:
            print("Initializing E5 embedding service for context retrieval...")
            embed_service = E5EmbeddingService()
            print(f"RAG enabled: {len(chunks)} chunks loaded")
        else:
            print("Warning: No chunks loaded, proceeding without RAG grounding")
    else:
        print("Warning: RAG not available, proceeding without fact grounding")
    
    # Retrieve contexts for all prompts
    print("\nRetrieving RAG contexts for each question...")
    contexts = retrieve_contexts_batch(prompts, chunks, embed_service, top_k=config.rag_top_k)
    print(f"Retrieved contexts for {len(contexts)} questions")
    
    judge = VLLMJudge(config.judge_model_name, config)
    judge.load()
    
    # Score BASE model responses with RAG context
    print("\nScoring BASE model responses (with RAG grounding)...")
    base_scores = []
    num_batches = (len(prompts) + config.judge_batch_size - 1) // config.judge_batch_size
    
    for i in tqdm(range(num_batches), desc="Judging base model"):
        start = i * config.judge_batch_size
        end = min(start + config.judge_batch_size, len(prompts))
        batch_scores = judge.score_batch(
            prompts[start:end],
            base_responses[start:end],
            contexts[start:end]  # Pass RAG contexts
        )
        base_scores.extend(batch_scores)
    
    # Score FINETUNED model responses with RAG context
    print("\nScoring FINETUNED model responses (with RAG grounding)...")
    ft_scores = []
    
    for i in tqdm(range(num_batches), desc="Judging finetuned model"):
        start = i * config.judge_batch_size
        end = min(start + config.judge_batch_size, len(prompts))
        batch_scores = judge.score_batch(
            prompts[start:end],
            ft_responses[start:end],
            contexts[start:end]  # Pass RAG contexts
        )
        ft_scores.extend(batch_scores)

    
    # Free judge memory
    del judge
    torch.cuda.empty_cache()
    
    # ===========================================
    # Compute Statistics
    # ===========================================
    def compute_stats(scores: List[Dict]) -> Dict:
        """Compute average scores per dimension (6 criteria)."""
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
    
    results["base_model"] = {
        "stats": base_stats,
        "sample_responses": base_responses[:5],
        "sample_scores": base_scores[:5],
    }
    
    results["finetuned_model"] = {
        "stats": ft_stats,
        "sample_responses": ft_responses[:5],
        "sample_scores": ft_scores[:5],
    }
    
    # Comparison (6 criteria)
    comparison = {}
    for dim in ["helpfulness", "relevance", "factual_accuracy", "hallucination_check", "coherence", "fluency", "total"]:
        base_mean = base_stats.get(dim, {}).get("mean", 0)
        ft_mean = ft_stats.get(dim, {}).get("mean", 0)
        improvement = ft_mean - base_mean
        pct_improvement = (improvement / base_mean * 100) if base_mean > 0 else 0
        
        comparison[dim] = {
            "base": base_mean,
            "finetuned": ft_mean,
            "improvement": improvement,
            "improvement_percent": pct_improvement
        }
    
    results["comparison"] = comparison
    
    # Detailed results - include category and all responses
    for i, (sample, base_resp, ft_resp, base_sc, ft_sc) in enumerate(
        zip(test_samples, base_responses, ft_responses, base_scores, ft_scores)
    ):
        results["detailed_results"].append({
            "id": sample["id"],
            "category": sample.get("category", ""),
            "prompt": sample["prompt"],
            "expected_response": sample.get("expected", ""),
            "base_model_response": base_resp,
            "finetuned_model_response": ft_resp,
            "base_model_scores": base_sc,
            "finetuned_model_scores": ft_sc,
            "score_comparison": {
                "base_total": base_sc.get("total", 0),
                "finetuned_total": ft_sc.get("total", 0),
                "winner": "finetuned" if ft_sc.get("total", 0) > base_sc.get("total", 0) else ("base" if base_sc.get("total", 0) > ft_sc.get("total", 0) else "tie")
            }
        })
    
    # ===========================================
    # Save Results
    # ===========================================
    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    with open(config.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # ===========================================
    # Print Summary
    # ===========================================
    print("\n" + "="*75)
    print("RAG-GROUNDED EVALUATION RESULTS - 6-POINT CRITERIA")
    print("="*75)
    print(f"\n{'Dimension':<20} {'Base Model':>12} {'Finetuned':>12} {'Improvement':>12}")
    print("-"*75)
    
    for dim in ["helpfulness", "relevance", "factual_accuracy", "hallucination_check", "coherence", "fluency"]:
        base_mean = base_stats.get(dim, {}).get("mean", 0)
        ft_mean = ft_stats.get(dim, {}).get("mean", 0)
        improvement = ft_mean - base_mean
        label = dim.replace("_", " ").title()
        print(f"{label:<20} {base_mean:>12.2f} {ft_mean:>12.2f} {improvement:>+12.2f}")
    
    print("-"*75)
    base_total = base_stats.get("total", {}).get("mean", 0)
    ft_total = ft_stats.get("total", {}).get("mean", 0)
    total_improvement = ft_total - base_total
    pct = (total_improvement / base_total * 100) if base_total > 0 else 0
    
    print(f"{'OVERALL SCORE':<20} {base_total:>12.2f} {ft_total:>12.2f} {total_improvement:>+9.2f} ({pct:+.1f}%)")
    print("="*75)
    print(f"\nResults saved to: {config.output_path}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-Judge Evaluation")
    parser.add_argument("--base_model", type=str, default="google/gemma-3-1b-it",
                        help="Base model name")
    parser.add_argument("--finetuned_path", type=str, 
                        default="./outputs/gemma3-1b-qlora-sft/final_model",
                        help="Path to fine-tuned model")
    parser.add_argument("--judge_model", type=str, default="google/gemma-3-12b-it",
                        help="Judge model name")
    parser.add_argument("--test_dataset", type=str,
                        default="../data/final/multiturn_dataset_cleaned_no_thought.json",
                        help="Path to test dataset")
    parser.add_argument("--output", type=str, 
                        default="./outputs/llm_judge_evaluation_results.json",
                        help="Path to save results")
    parser.add_argument("--gen_batch_size", type=int, default=16,
                        help="Batch size for response generation")
    parser.add_argument("--judge_batch_size", type=int, default=32,
                        help="Batch size for judging")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Maximum number of test samples")
    
    args = parser.parse_args()
    
    config = EvalConfig(
        base_model_name=args.base_model,
        finetuned_model_path=args.finetuned_path,
        judge_model_name=args.judge_model,
        test_dataset_path=args.test_dataset,
        output_path=args.output,
        generation_batch_size=args.gen_batch_size,
        judge_batch_size=args.judge_batch_size,
        max_test_samples=args.max_samples,
    )
    
    evaluate(config)
