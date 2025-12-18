# QLoRA SFT Fine-tuning for Gemma 3-1B-IT

Folder ini berisi script untuk fine-tuning model **Gemma 3-1B-IT** menggunakan metode **QLoRA + SFT**.
Optimized untuk **A100 80GB** dengan best practices 2024.

## 📁 Struktur Folder

```
finetuning/
├── train_qlora_sft.py      # Main training script
├── evaluate_bertscore.py   # BERTScore evaluation (base vs fine-tuned)
├── inference.py            # Inference dengan model hasil training
├── run_training.sh         # Script untuk training
├── run_evaluation.sh       # Script untuk evaluation
├── requirements.txt        # Dependencies
└── README.md               # Dokumentasi ini
```

## 🎯 Best Practices yang Diterapkan

### QLoRA Configuration
| Parameter | Value | Referensi |
|-----------|-------|-----------|
| LoRA Rank (r) | 32 | [HuggingFace PEFT Guide] |
| LoRA Alpha | 64 (2x rank) | Standard scaling |
| Target Modules | q,k,v,o_proj + gate,up,down_proj | Comprehensive adaptation |
| Dropout | 0.05 | Prevent overfitting |
| Quantization | 4-bit NF4 + double quant | [QLoRA Paper] |

### Training Configuration
| Parameter | Value | Keterangan |
|-----------|-------|------------|
| Model | google/gemma-3-1b-it | 1B model untuk training lebih cepat |
| Batch Size | 8 x 8 = 64 | per_device x gradient_accum |
| Learning Rate | 2e-4 | Standard untuk LoRA |
| Scheduler | Cosine with 10% warmup | Best for fine-tuning |
| Precision | bfloat16 | Optimal untuk A100 |
| NEFTune Alpha | 5.0 | Improve generalization |
| Flash Attention | Enabled | Speed optimization |

### BERTScore Evaluation
- **Batch processing** untuk kecepatan
- Membandingkan **base model** vs **fine-tuned model**
- Menggunakan **DeBERTa-XLarge-MNLI** untuk semantic similarity

## 🚀 Cara Penggunaan

### 1. Setup Environment
```bash
cd finetuning
source ../venv/bin/activate

# Install additional dependencies if needed
pip install datasets
```

### 2. Jalankan Training
```bash
chmod +x run_training.sh
./run_training.sh
```

Atau dengan custom parameters:
```bash
python train_qlora_sft.py \
    --dataset ../data/final/multiturn_dataset_cleaned.json \
    --output_dir ./outputs/gemma3-1b-qlora-sft \
    --epochs 3 \
    --batch_size 8 \
    --grad_accum 8 \
    --lr 2e-4 \
    --lora_r 32 \
    --lora_alpha 64
```

### 3. Jalankan Evaluation
```bash
chmod +x run_evaluation.sh
./run_evaluation.sh
```

Atau manual:
```bash
python evaluate_bertscore.py \
    --base_model google/gemma-3-1b-it \
    --finetuned_path ./outputs/gemma3-1b-qlora-sft/final_model \
    --batch_size 16 \
    --max_samples 100
```

### 4. Inference
```bash
# Interactive mode
python inference.py \
    --model_path ./outputs/gemma3-1b-qlora-sft/final_model \
    --interactive

# Single prompt
python inference.py \
    --model_path ./outputs/gemma3-1b-qlora-sft/final_model \
    --prompt "Berapa biaya kuliah di UNSIQ?"
```

## 📊 Output Evaluation

Hasil evaluation akan tersimpan di `./outputs/evaluation_results.json`:

```json
{
  "base_model": {
    "bertscore": {"precision": 0.xx, "recall": 0.xx, "f1": 0.xx}
  },
  "finetuned_model": {
    "bertscore": {"precision": 0.xx, "recall": 0.xx, "f1": 0.xx}
  },
  "comparison": {
    "f1_improvement": 0.xx,
    "f1_improvement_percent": xx.x
  }
}
```

## 💾 Hardware Requirements

| GPU | Batch Size | Training Time (1000 samples, 3 epochs) |
|-----|------------|---------------------------------------|
| A100 80GB | 8 | ~30 menit |
| A100 40GB | 4 | ~45 menit |
| A10 24GB | 2 | ~1.5 jam |

## 📚 Referensi

- [QLoRA Paper](https://arxiv.org/abs/2305.14314) - Dettmers et al. 2023
- [HuggingFace PEFT](https://huggingface.co/docs/peft)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [Gemma Fine-tuning Guide](https://ai.google.dev/gemma/docs/fine_tuning)
