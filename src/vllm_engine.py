"""
vLLM Engine for High-Performance Generation
Optimized for A100 80GB with Gemma-3-12B

Features:
- PagedAttention for efficient memory management
- Continuous batching for high throughput
- Batch inference for dataset generation
- LLM-as-Judge scoring
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    print("Warning: vLLM not installed. Install with: pip install vllm")


# =============================================================================
# A100 80GB OPTIMIZED SETTINGS
# =============================================================================

A100_80GB_CONFIG = {
    # Model settings
    "model_name": "google/gemma-3-12b-it",
    "dtype": "bfloat16",  # Best for A100
    "trust_remote_code": True,
    
    # Memory settings
    "gpu_memory_utilization": 0.90,  # 90% of 80GB = 72GB available
    "max_model_len": 4096,  # Max sequence length
    
    # Tensor parallelism (set to number of GPUs)
    "tensor_parallel_size": 1,  # Single A100
    
    # Batch settings for high throughput
    "max_num_batched_tokens": 16384,  # Large batch for A100
    "max_num_seqs": 256,  # Max concurrent sequences
    
    # KV Cache optimization
    "block_size": 16,  # PagedAttention block size
    "swap_space": 4,  # GB for CPU swap
    
    # Generation defaults
    "default_max_tokens": 512,
    "default_temperature": 0.8,
    "default_top_p": 0.95,
}

# For multi-GPU setup (2x A100)
A100_MULTI_GPU_CONFIG = {
    **A100_80GB_CONFIG,
    "tensor_parallel_size": 2,
    "max_num_batched_tokens": 32768,
    "max_num_seqs": 512,
}


# =============================================================================
# VLLM ENGINE WRAPPER
# =============================================================================

class VLLMEngine:
    """
    vLLM Engine wrapper for high-performance batch generation.
    Optimized for A100 80GB.
    """
    
    def __init__(self, model_name: str = None, config: Dict[str, Any] = None,
                 multi_gpu: bool = False):
        """
        Initialize vLLM engine.
        
        Args:
            model_name: HuggingFace model name (default: Gemma-3-12B-IT)
            config: Custom configuration (default: A100 optimized)
            multi_gpu: Use multi-GPU config if True
        """
        if not HAS_VLLM:
            raise ImportError("vLLM is not installed. Install with: pip install vllm")
        
        self.config = config or (A100_MULTI_GPU_CONFIG if multi_gpu else A100_80GB_CONFIG)
        self.model_name = model_name or self.config["model_name"]
        self.llm = None
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """Load the model into GPU memory."""
        if self.is_loaded:
            return True
        
        print(f"Loading model: {self.model_name}")
        print(f"Configuration: dtype={self.config['dtype']}, "
              f"tensor_parallel={self.config['tensor_parallel_size']}, "
              f"gpu_memory={self.config['gpu_memory_utilization']}")
        
        try:
            self.llm = LLM(
                model=self.model_name,
                dtype=self.config["dtype"],
                trust_remote_code=self.config["trust_remote_code"],
                gpu_memory_utilization=self.config["gpu_memory_utilization"],
                max_model_len=self.config["max_model_len"],
                tensor_parallel_size=self.config["tensor_parallel_size"],
                max_num_batched_tokens=self.config.get("max_num_batched_tokens"),
                max_num_seqs=self.config.get("max_num_seqs"),
                block_size=self.config.get("block_size", 16),
                swap_space=self.config.get("swap_space", 4),
            )
            self.is_loaded = True
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def generate_batch(self, prompts: List[str],
                       max_tokens: int = None,
                       temperature: float = None,
                       top_p: float = None,
                       stop_sequences: List[str] = None) -> List[str]:
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop_sequences: Sequences to stop generation
            
        Returns:
            List of generated responses
        """
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError("Failed to load model")
        
        # Use config defaults if not specified
        max_tokens = max_tokens or self.config["default_max_tokens"]
        temperature = temperature if temperature is not None else self.config["default_temperature"]
        top_p = top_p or self.config["default_top_p"]
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_sequences or [],
        )
        
        # Batch generation
        outputs = self.llm.generate(prompts, sampling_params)
        
        # Extract generated text
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)
        
        return results
    
    def generate_single(self, prompt: str, **kwargs) -> str:
        """Generate response for a single prompt."""
        results = self.generate_batch([prompt], **kwargs)
        return results[0] if results else ""


# =============================================================================
# DATASET GENERATION WITH VLLM
# =============================================================================

class VLLMDatasetGenerator:
    """
    Generate instruction-output pairs using vLLM for high throughput.
    """
    
    # Enhanced System Prompt with Factual Grounding (Layer 2)
    # Based on: RAGTruth (arXiv 2024), RAG-HAT (ACL 2024)
    GENERATION_SYSTEM_PROMPT = """Anda adalah asisten AI untuk PMB UNSIQ yang HANYA menjawab berdasarkan konteks.

FAKTA WAJIB DIGUNAKAN (JANGAN DIUBAH):
- Nama Institusi: Universitas Sains Al-Qur'an (UNSIQ)
- Lokasi: Jl. KH. Hasyim Asy'ari Km. 03, Kalibebar, Kec. Mojotengah, Kab. Wonosobo, Jawa Tengah 56351
- Moto: Kuliah Plus Ngaji
- Visi: Menjadi universitas unggul berbasis Al-Qur'an dan sains

ATURAN KETAT - WAJIB DIPATUHI:
1. HANYA gunakan informasi yang ADA dalam konteks
2. Jika informasi TIDAK ADA dalam konteks, JANGAN mengarang - tulis "Informasi tidak tersedia"
3. JANGAN ubah nama institusi, fakultas, atau program studi
4. Mulai jawaban dengan "Berdasarkan informasi PMB UNSIQ..."
5. Gunakan format bullet points untuk kejelasan

FAKULTAS RESMI UNSIQ (gunakan nama persis ini):
- Fakultas Teknik dan Ilmu Komputer (FASTIKOM)
- Fakultas Ilmu Kesehatan (FIKES)
- Fakultas Ekonomi dan Bisnis
- Fakultas Syariah dan Hukum (FSH)
- Fakultas Ilmu Tarbiyah dan Keguruan (FITK)
- Fakultas Komunikasi dan Sosial Politik (FKSP)
- Fakultas Bahasa dan Sastra

Format output:
PERTANYAAN: [pertanyaan natural dari calon mahasiswa]
JAWABAN: [jawaban lengkap berdasarkan konteks]"""

    GENERATION_USER_TEMPLATE = """KONTEKS SUMBER (gunakan HANYA informasi ini):
{context}

Jenis pertanyaan: {question_type}
Topik: {topic}

INSTRUKSI: Buat 1 pasangan pertanyaan-jawaban yang AKURAT berdasarkan konteks di atas.
- Pertanyaan harus natural seperti calon mahasiswa bertanya
- Jawaban WAJIB berdasarkan konteks, JANGAN mengarang
- Jika konteks tidak lengkap, jawab sesuai yang tersedia saja"""

    def __init__(self, engine: VLLMEngine):
        self.engine = engine
        
    def _create_generation_prompt(self, chunk: Dict, question_type: str) -> str:
        """Create prompt for generating Q&A pair."""
        context = chunk.get("content", "")
        topic = chunk.get("topic", "umum")
        
        user_prompt = self.GENERATION_USER_TEMPLATE.format(
            context=context,
            question_type=question_type,
            topic=topic
        )
        
        # Format for Gemma chat
        prompt = f"""<start_of_turn>user
{self.GENERATION_SYSTEM_PROMPT}

{user_prompt}<end_of_turn>
<start_of_turn>model
"""
        return prompt
    
    def _parse_generated_output(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse generated text to extract question and answer."""
        import re
        
        # Try to find PERTANYAAN/JAWABAN pattern
        q_match = re.search(r'PERTANYAAN:\s*(.+?)(?=JAWABAN:|$)', text, re.DOTALL | re.IGNORECASE)
        a_match = re.search(r'JAWABAN:\s*(.+?)$', text, re.DOTALL | re.IGNORECASE)
        
        if q_match and a_match:
            question = q_match.group(1).strip()
            answer = a_match.group(1).strip()
            return question, answer
        
        return None, None
    
    def generate_pairs(self, chunks: List[Dict], 
                       question_types: List[str] = None,
                       batch_size: int = 64,
                       max_tokens: int = 512,
                       pairs_per_chunk: int = 43) -> List[Dict]:
        """
        Generate instruction-output pairs from chunks using vLLM batch inference.
        
        Args:
            chunks: List of chunk dictionaries
            question_types: Types of questions to generate
            batch_size: Number of prompts per batch
            max_tokens: Max tokens for generation
            pairs_per_chunk: Number of pairs to generate per chunk (default 43 for ~7000 total)
            
        Returns:
            List of generated pairs
        """
        if question_types is None:
            question_types = ["faktual", "prosedural", "komparatif", "kondisional", "eksploratif"]
        
        # Create all prompts - generate pairs_per_chunk variations per chunk
        all_prompts = []
        prompt_metadata = []
        
        # Temperature variations for diversity
        temperatures = [0.7, 0.8, 0.9, 1.0]
        
        for chunk in chunks:
            for pair_idx in range(pairs_per_chunk):
                # Cycle through question types
                q_type = question_types[pair_idx % len(question_types)]
                # Vary temperature for diversity
                temp = temperatures[pair_idx % len(temperatures)]
                
                prompt = self._create_generation_prompt(chunk, q_type)
                all_prompts.append(prompt)
                prompt_metadata.append({
                    "chunk_id": chunk.get("id", "unknown"),
                    "topic": chunk.get("topic", "umum"),
                    "question_type": q_type,
                    "temperature": temp,
                    "variation_idx": pair_idx
                })
        
        print(f"Generating {len(all_prompts)} prompts in batches of {batch_size}...")
        print(f"  - Chunks: {len(chunks)}")
        print(f"  - Pairs per chunk: {pairs_per_chunk}")
        print(f"  - Expected pairs: ~{len(chunks) * pairs_per_chunk}")
        
        # Process in batches
        all_pairs = []
        total_batches = (len(all_prompts) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(all_prompts), batch_size):
            batch_prompts = all_prompts[batch_idx:batch_idx + batch_size]
            batch_meta = prompt_metadata[batch_idx:batch_idx + batch_size]
            
            batch_num = batch_idx // batch_size + 1
            print(f"Processing batch {batch_num}/{total_batches}... ({len(all_pairs)} pairs so far)")
            
            # Use temperature from first item in batch (could optimize per-prompt later)
            batch_temp = batch_meta[0].get("temperature", 0.8)
            
            # Generate
            outputs = self.engine.generate_batch(
                batch_prompts,
                max_tokens=max_tokens,
                temperature=batch_temp,
                stop_sequences=["<end_of_turn>"]
            )
            
            # Parse outputs
            for output, meta in zip(outputs, batch_meta):
                question, answer = self._parse_generated_output(output)
                
                if question and answer:
                    all_pairs.append({
                        "instruction": question,
                        "output": answer,
                        "source_chunk_id": meta["chunk_id"],
                        "topic": meta["topic"],
                        "question_type": meta["question_type"],
                        "generation_mode": "vllm",
                        "variation_idx": meta["variation_idx"]
                    })
        
        print(f"Generated {len(all_pairs)} valid pairs from {len(chunks)} chunks")
        return all_pairs


# =============================================================================
# LLM-AS-JUDGE WITH VLLM
# =============================================================================

class VLLMLLMJudge:
    """
    LLM-as-Judge implementation using vLLM for high-throughput scoring.
    """
    
    JUDGE_SYSTEM_PROMPT = """Anda adalah penilai kualitas dataset yang ketat dan objektif.

Evaluasi pasangan pertanyaan-jawaban berdasarkan kriteria berikut:
1. Akurasi (0-10): Apakah jawaban faktual dan benar?
2. Kelengkapan (0-10): Apakah jawaban lengkap dan tidak ada informasi penting yang hilang?
3. Kejelasan (0-10): Apakah jawaban jelas dan mudah dipahami?
4. Relevansi (0-10): Apakah jawaban relevan dengan pertanyaan?
5. Format (0-10): Apakah format jawaban rapi dan terstruktur?

Berikan penilaian dalam format JSON:
{
  "akurasi": <skor>,
  "kelengkapan": <skor>,
  "kejelasan": <skor>,
  "relevansi": <skor>,
  "format": <skor>,
  "total": <rata-rata>,
  "komentar": "<komentar singkat>",
  "rekomendasi": "TERIMA" | "TOLAK" | "REVISI"
}"""

    JUDGE_USER_TEMPLATE = """Konteks Referensi:
{context}

Pertanyaan:
{instruction}

Jawaban:
{output}

Berikan penilaian dalam format JSON."""

    def __init__(self, engine: VLLMEngine):
        self.engine = engine
        
    def _create_judge_prompt(self, pair: Dict, context: str = "") -> str:
        """Create prompt for judging a pair."""
        user_prompt = self.JUDGE_USER_TEMPLATE.format(
            context=context or "Tidak ada konteks tambahan",
            instruction=pair.get("instruction", ""),
            output=pair.get("output", "")
        )
        
        prompt = f"""<start_of_turn>user
{self.JUDGE_SYSTEM_PROMPT}

{user_prompt}<end_of_turn>
<start_of_turn>model
"""
        return prompt
    
    def _parse_judge_output(self, text: str) -> Optional[Dict]:
        """Parse judge output to extract scores."""
        import re
        
        # Try to find JSON in the output
        json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        
        if json_match:
            try:
                scores = json.loads(json_match.group())
                # Validate required fields
                required = ["akurasi", "kelengkapan", "kejelasan", "relevansi", "format"]
                if all(k in scores for k in required):
                    # Calculate total if not present
                    if "total" not in scores:
                        scores["total"] = sum(scores[k] for k in required) / len(required)
                    return scores
            except json.JSONDecodeError:
                pass
        
        return None
    
    def judge_pairs(self, pairs: List[Dict],
                    contexts: Dict[str, str] = None,
                    batch_size: int = 64,
                    max_tokens: int = 256) -> List[Dict]:
        """
        Judge a list of pairs using LLM-as-Judge.
        
        Args:
            pairs: List of instruction-output pairs
            contexts: Optional dict mapping chunk_id to context
            batch_size: Number of prompts per batch
            max_tokens: Max tokens for judge output
            
        Returns:
            List of pairs with scores added
        """
        contexts = contexts or {}
        
        # Create judge prompts
        all_prompts = []
        for pair in pairs:
            chunk_id = pair.get("source_chunk_id", "")
            context = contexts.get(chunk_id, "")
            prompt = self._create_judge_prompt(pair, context)
            all_prompts.append(prompt)
        
        print(f"Judging {len(all_prompts)} pairs in batches of {batch_size}...")
        
        # Process in batches
        all_scores = []
        total_batches = (len(all_prompts) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(all_prompts), batch_size):
            batch_prompts = all_prompts[batch_idx:batch_idx + batch_size]
            
            batch_num = batch_idx // batch_size + 1
            print(f"Processing batch {batch_num}/{total_batches}...")
            
            outputs = self.engine.generate_batch(
                batch_prompts,
                max_tokens=max_tokens,
                temperature=0.1,  # Low temperature for consistent scoring
                stop_sequences=["<end_of_turn>"]
            )
            
            for output in outputs:
                scores = self._parse_judge_output(output)
                all_scores.append(scores)
        
        # Combine pairs with scores
        scored_pairs = []
        for pair, scores in zip(pairs, all_scores):
            pair_copy = pair.copy()
            if scores:
                pair_copy["llm_judge_scores"] = scores
                pair_copy["llm_judge_total"] = scores.get("total", 0)
                pair_copy["llm_judge_recommendation"] = scores.get("rekomendasi", "UNKNOWN")
            else:
                pair_copy["llm_judge_scores"] = None
                pair_copy["llm_judge_total"] = 0
                pair_copy["llm_judge_recommendation"] = "ERROR"
            scored_pairs.append(pair_copy)
        
        # Summary
        accepted = sum(1 for p in scored_pairs if p.get("llm_judge_recommendation") == "TERIMA")
        rejected = sum(1 for p in scored_pairs if p.get("llm_judge_recommendation") == "TOLAK")
        revision = sum(1 for p in scored_pairs if p.get("llm_judge_recommendation") == "REVISI")
        errors = sum(1 for p in scored_pairs if p.get("llm_judge_recommendation") == "ERROR")
        
        print(f"\nJudging complete:")
        print(f"  TERIMA: {accepted}")
        print(f"  TOLAK: {rejected}")
        print(f"  REVISI: {revision}")
        print(f"  ERROR: {errors}")
        
        return scored_pairs


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_chunks(chunks_path: str) -> List[Dict]:
    """Load chunks from JSONL file."""
    chunks = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def save_pairs(pairs: List[Dict], output_path: str):
    """Save pairs to JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')


def run_vllm_generation(chunks_path: str,
                        output_path: str,
                        model_name: str = None,
                        batch_size: int = 64,
                        multi_gpu: bool = False,
                        pairs_per_chunk: int = 43) -> Dict[str, Any]:
    """
    Run full vLLM generation pipeline.
    
    Args:
        chunks_path: Path to chunks JSONL file
        output_path: Path for output pairs
        model_name: Model name (default: Gemma-3-12B-IT)
        batch_size: Batch size for generation
        multi_gpu: Use multi-GPU configuration
        pairs_per_chunk: Number of pairs per chunk (default 43 for ~7000 total)
        
    Returns:
        Results dictionary
    """
    start_time = datetime.now()
    
    # Load chunks
    print(f"Loading chunks from {chunks_path}...")
    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks")
    print(f"Target: {len(chunks)} × {pairs_per_chunk} = ~{len(chunks) * pairs_per_chunk} pairs")
    
    # Initialize engine
    print("\nInitializing vLLM engine...")
    engine = VLLMEngine(model_name=model_name, multi_gpu=multi_gpu)
    
    # Generate pairs
    generator = VLLMDatasetGenerator(engine)
    pairs = generator.generate_pairs(chunks, batch_size=batch_size, pairs_per_chunk=pairs_per_chunk)
    
    # Save pairs
    save_pairs(pairs, output_path)
    print(f"\nSaved {len(pairs)} pairs to {output_path}")
    
    duration = (datetime.now() - start_time).total_seconds()
    
    results = {
        "chunks_processed": len(chunks),
        "pairs_generated": len(pairs),
        "duration_seconds": duration,
        "pairs_per_second": len(pairs) / duration if duration > 0 else 0,
        "output_file": output_path
    }
    
    return results


def run_vllm_judge(pairs_path: str,
                   output_path: str,
                   chunks_path: str = None,
                   model_name: str = None,
                   batch_size: int = 64,
                   multi_gpu: bool = False) -> Dict[str, Any]:
    """
    Run LLM-as-Judge on pairs using vLLM.
    
    Args:
        pairs_path: Path to pairs JSONL file
        output_path: Path for scored pairs output
        chunks_path: Optional path to chunks for context
        model_name: Judge model name
        batch_size: Batch size for judging
        multi_gpu: Use multi-GPU configuration
        
    Returns:
        Results dictionary
    """
    start_time = datetime.now()
    
    # Load pairs
    print(f"Loading pairs from {pairs_path}...")
    pairs = load_chunks(pairs_path)  # Same format as chunks
    print(f"Loaded {len(pairs)} pairs")
    
    # Load contexts if available
    contexts = {}
    if chunks_path and os.path.exists(chunks_path):
        chunks = load_chunks(chunks_path)
        contexts = {c.get("id", ""): c.get("content", "") for c in chunks}
        print(f"Loaded {len(contexts)} context chunks")
    
    # Initialize engine
    print("\nInitializing vLLM engine for judging...")
    engine = VLLMEngine(model_name=model_name, multi_gpu=multi_gpu)
    
    # Judge pairs
    judge = VLLMLLMJudge(engine)
    scored_pairs = judge.judge_pairs(pairs, contexts, batch_size=batch_size)
    
    # Save scored pairs
    save_pairs(scored_pairs, output_path)
    print(f"\nSaved {len(scored_pairs)} scored pairs to {output_path}")
    
    duration = (datetime.now() - start_time).total_seconds()
    
    # Calculate statistics
    valid_scores = [p for p in scored_pairs if p.get("llm_judge_scores")]
    avg_score = sum(p.get("llm_judge_total", 0) for p in valid_scores) / len(valid_scores) if valid_scores else 0
    
    results = {
        "pairs_judged": len(pairs),
        "valid_scores": len(valid_scores),
        "average_score": avg_score,
        "accepted": sum(1 for p in scored_pairs if p.get("llm_judge_recommendation") == "TERIMA"),
        "rejected": sum(1 for p in scored_pairs if p.get("llm_judge_recommendation") == "TOLAK"),
        "revision": sum(1 for p in scored_pairs if p.get("llm_judge_recommendation") == "REVISI"),
        "duration_seconds": duration,
        "pairs_per_second": len(pairs) / duration if duration > 0 else 0,
        "output_file": output_path
    }
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if not HAS_VLLM:
        print("vLLM is not installed. Install with:")
        print("  pip install vllm")
        sys.exit(1)
    
    # Example usage
    mode = sys.argv[1] if len(sys.argv) > 1 else "help"
    
    if mode == "generate":
        # Generate pairs from chunks
        chunks_path = sys.argv[2] if len(sys.argv) > 2 else "data/chunks/chunks.jsonl"
        output_path = sys.argv[3] if len(sys.argv) > 3 else "data/raw/dataset_vllm.jsonl"
        
        results = run_vllm_generation(chunks_path, output_path)
        print(f"\n=== Generation Results ===")
        print(f"Pairs generated: {results['pairs_generated']}")
        print(f"Duration: {results['duration_seconds']:.1f}s")
        print(f"Speed: {results['pairs_per_second']:.1f} pairs/sec")
        
    elif mode == "judge":
        # Judge existing pairs
        pairs_path = sys.argv[2] if len(sys.argv) > 2 else "data/raw/dataset_raw.jsonl"
        output_path = sys.argv[3] if len(sys.argv) > 3 else "data/filtered/dataset_judged.jsonl"
        chunks_path = sys.argv[4] if len(sys.argv) > 4 else "data/chunks/chunks.jsonl"
        
        results = run_vllm_judge(pairs_path, output_path, chunks_path)
        print(f"\n=== Judging Results ===")
        print(f"Pairs judged: {results['pairs_judged']}")
        print(f"Average score: {results['average_score']:.2f}/10")
        print(f"Accepted: {results['accepted']}")
        print(f"Rejected: {results['rejected']}")
        print(f"Duration: {results['duration_seconds']:.1f}s")
        
    else:
        print("vLLM Dataset Generator for A100 80GB")
        print("=" * 50)
        print("\nUsage:")
        print("  python vllm_engine.py generate [chunks_path] [output_path]")
        print("  python vllm_engine.py judge [pairs_path] [output_path] [chunks_path]")
        print("\nConfiguration (A100 80GB):")
        for key, value in A100_80GB_CONFIG.items():
            print(f"  {key}: {value}")
