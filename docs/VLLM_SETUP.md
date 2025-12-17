# vLLM Integration for A100 80GB

## Overview

This module provides high-performance inference using vLLM, optimized for NVIDIA A100 80GB GPU.

## Features

- **PagedAttention**: Efficient memory management for KV cache
- **Continuous Batching**: Dynamic batch updates for maximum throughput
- **Batch Inference**: Process 64+ prompts simultaneously
- **LLM-as-Judge**: Automated quality scoring at scale

## A100 80GB Configuration

| Setting | Value | Description |
|---------|-------|-------------|
| `gpu_memory_utilization` | 90% | ~72GB for model |
| `dtype` | bfloat16 | Best precision/speed balance |
| `max_model_len` | 4096 | Max sequence length |
| `max_num_batched_tokens` | 16384 | High throughput batch size |
| `max_num_seqs` | 256 | Max concurrent sequences |
| `block_size` | 16 | PagedAttention block size |

## Installation

```bash
# Requires CUDA 12.1+
pip install vllm>=0.6.0
```

## Usage

### Quick Start
```bash
python run_vllm.py
```

### Dataset Generation Only
```bash
python src/vllm_engine.py generate data/chunks/chunks.jsonl data/raw/dataset_vllm.jsonl
```

### LLM-as-Judge Only
```bash
python src/vllm_engine.py judge data/raw/dataset_raw.jsonl data/filtered/dataset_judged.jsonl
```

### Python API
```python
from src.vllm_engine import VLLMEngine, VLLMDatasetGenerator, VLLMLLMJudge

# Initialize engine
engine = VLLMEngine(model_name="google/gemma-3-12b-it")

# Generate pairs
generator = VLLMDatasetGenerator(engine)
pairs = generator.generate_pairs(chunks, batch_size=64)

# Judge pairs
judge = VLLMLLMJudge(engine)
scored = judge.judge_pairs(pairs, batch_size=64)
```

## Expected Performance

| Task | A100 80GB | With vLLM |
|------|-----------|-----------|
| Generation (1000 pairs) | ~30 min (HF) | **~2 min** |
| Judging (1000 pairs) | ~30 min | **~3 min** |
| Total throughput | ~1 pair/sec | **~10+ pairs/sec** |

## Multi-GPU Setup (2x A100)

Edit `config/config.yaml`:
```yaml
vllm:
  tensor_parallel_size: 2
  max_num_batched_tokens: 32768
  max_num_seqs: 512
```

Or use in Python:
```python
engine = VLLMEngine(multi_gpu=True)
```

## Files

- `src/vllm_engine.py` - Main vLLM wrapper and utilities
- `run_vllm.py` - Pipeline script for generation + judging
- `config/config.yaml` - Configuration under `vllm:` section

## Troubleshooting

### Out of Memory
- Reduce `gpu_memory_utilization` to 0.85
- Reduce `max_num_batched_tokens` to 8192
- Reduce batch_size to 32

### Slow Startup
- First run downloads model (~25GB for Gemma-3-12B)
- Subsequent runs use cached model

### CUDA Version
- vLLM requires CUDA 12.1+
- Check with: `nvcc --version`
