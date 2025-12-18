"""
QLoRA SFT Fine-tuning Script for Gemma 3-1B-IT
Using TRL SFTTrainer with PEFT QLoRA configuration
Optimized for A100 80GB - Best Practices 2024

References:
- https://huggingface.co/docs/trl/sft_trainer
- https://huggingface.co/docs/peft/task_guides/lora
- QLoRA Paper: Dettmers et al. 2023
"""

import os
import json
import torch
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainerCallback,
    EarlyStoppingCallback,
)
from transformers.integrations import TensorBoardCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Version-aware import for TRL config
try:
    from trl import SFTConfig
    USE_SFT_CONFIG = True
except ImportError:
    from transformers import TrainingArguments as SFTConfig
    USE_SFT_CONFIG = False

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
    """Setup comprehensive logging to file and console."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger("qlora_sft")
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

# =============================================================================
# CONFIGURATION - Best Practices 2024
# =============================================================================

@dataclass
class ModelConfig:
    """Model configuration for Gemma 3-1B-IT."""
    model_name: str = "google/gemma-3-1b-it"  # 1B model for faster training
    max_seq_length: int = 2048  # Optimal for multi-turn conversations
    dtype: torch.dtype = torch.bfloat16  # Best for A100

@dataclass  
class LoRAConfig:
    """LoRA configuration following best practices.
    
    Best Practices:
    - rank (r): 16-32 for good balance between performance and efficiency
    - alpha: typically 2x rank for stable training
    - dropout: 0.05-0.1 to prevent overfitting
    - target_modules: all attention + MLP layers for comprehensive adaptation
    """
    r: int = 32  # LoRA rank - higher for larger datasets
    lora_alpha: int = 64  # 2x rank for stable scaling
    lora_dropout: float = 0.05  # Prevent overfitting
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"  # MLP
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

@dataclass
class TrainConfig:
    """Training configuration optimized for A100 80GB.
    
    Best Practices:
    - Large effective batch size (256-512) via gradient accumulation
    - Cosine scheduler with warmup
    - bfloat16 for A100 stability
    - Gradient checkpointing for memory efficiency
    - NEFTune noise for better generalization
    """
    output_dir: str = "./outputs/gemma3-1b-qlora-sft"
    
    # Epochs & Steps
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 means use epochs
    
    # Batch Size - Optimized for A100 80GB with 1B model
    # Effective batch = per_device * accumulation = 8 * 8 = 64
    per_device_train_batch_size: int = 8  # Can be larger for 1B model
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8  # Effective batch = 64
    
    # Learning Rate Schedule
    learning_rate: float = 2e-4  # Standard for LoRA fine-tuning
    warmup_ratio: float = 0.1  # 10% warmup
    lr_scheduler_type: str = "cosine"  # Best for fine-tuning
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Precision & Memory
    fp16: bool = False
    bf16: bool = True  # Use bfloat16 for A100
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_32bit"  # Memory-efficient optimizer
    
    # Logging & Saving
    logging_steps: int = 10
    logging_first_step: bool = True
    logging_dir: str = None  # Will be set to output_dir/tensorboard
    save_steps: int = 100
    save_total_limit: int = 5  # Keep more checkpoints
    save_strategy: str = "steps"
    eval_strategy: str = "steps"
    eval_steps: int = 100
    
    # Early Stopping
    early_stopping_patience: int = 3  # Stop if no improvement for 3 evals
    early_stopping_threshold: float = 0.001
    
    # NEFTune - Add noise to embeddings for better generalization
    neftune_noise_alpha: float = 5.0  # Recommended value
    
    # Checkpoint settings
    save_safetensors: bool = True  # Use safetensors format
    resume_from_checkpoint: str = None  # Path to checkpoint to resume from
    
    # Misc
    seed: int = 42
    report_to: str = "tensorboard"  # Enable TensorBoard logging
    run_name: str = None  # Will be auto-generated if None

# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset_split(train_path: str, eval_path: str = None, eval_ratio: float = 0.1) -> tuple:
    """Load dataset from separate files or auto-split.
    
    Args:
        train_path: Path to training dataset (or full dataset if eval_path is None)
        eval_path: Optional path to evaluation dataset
        eval_ratio: Ratio for auto-split if eval_path is None
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    print(f"Loading training data from: {train_path}")
    
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    
    train_texts = [item["text"] for item in train_data if item.get("text")]
    print(f"Train examples: {len(train_texts)}")
    
    if eval_path and os.path.exists(eval_path):
        # Load separate eval file
        print(f"Loading eval data from: {eval_path}")
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        eval_texts = [item["text"] for item in eval_data if item.get("text")]
        print(f"Eval examples: {len(eval_texts)}")
    else:
        # Auto-split from train data
        print(f"Auto-splitting eval set with ratio {eval_ratio}")
        split_idx = int(len(train_texts) * (1 - eval_ratio))
        eval_texts = train_texts[split_idx:]
        train_texts = train_texts[:split_idx]
        print(f"After split - Train: {len(train_texts)}, Eval: {len(eval_texts)}")
    
    train_dataset = Dataset.from_dict({"text": train_texts})
    eval_dataset = Dataset.from_dict({"text": eval_texts})
    
    return train_dataset, eval_dataset

# =============================================================================
# MODEL SETUP
# =============================================================================

def setup_quantization_config(dtype: torch.dtype) -> BitsAndBytesConfig:
    """Setup 4-bit quantization config for QLoRA.
    
    Best Practices:
    - NF4 quantization type (optimal for normally distributed weights)
    - Double quantization for additional memory savings
    - bfloat16 compute dtype for A100
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # NormalFloat4 - best empirical results
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,  # Saves ~0.4 bits per param
    )

def setup_model_and_tokenizer(model_config: ModelConfig):
    """Setup model with QLoRA quantization and tokenizer."""
    print(f"\n{'='*60}")
    print(f"Loading model: {model_config.model_name}")
    print(f"{'='*60}")
    
    # Quantization config
    bnb_config = setup_quantization_config(model_config.dtype)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Right padding for training
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=model_config.dtype,
        attn_implementation="sdpa",  # Use PyTorch SDPA (no extra install needed)
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    
    # Print memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU Memory allocated: {allocated:.2f} GB")
    
    print("Model loaded with 4-bit quantization!")
    
    return model, tokenizer

def setup_lora(model, lora_config: LoRAConfig) -> tuple:
    """Apply LoRA configuration to the model."""
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
        task_type=lora_config.task_type,
    )
    
    model = get_peft_model(model, peft_config)
    
    print("\nLoRA Configuration:")
    print(f"  Rank (r): {lora_config.r}")
    print(f"  Alpha: {lora_config.lora_alpha}")
    print(f"  Dropout: {lora_config.lora_dropout}")
    print(f"  Target modules: {lora_config.target_modules}")
    
    model.print_trainable_parameters()
    
    return model, peft_config

# =============================================================================
# TRAINING
# =============================================================================

def train(
    dataset_path: str = "../data/final/multiturn_dataset_cleaned_no_thought.json",
    eval_dataset_path: str = None,
    model_config: ModelConfig = None,
    lora_config: LoRAConfig = None,
    train_config: TrainConfig = None,
):
    """Main training function with best practices, logging, and callbacks."""
    
    if model_config is None:
        model_config = ModelConfig()
    if lora_config is None:
        lora_config = LoRAConfig()
    if train_config is None:
        train_config = TrainConfig()
    
    # Create output directory
    os.makedirs(train_config.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(train_config.output_dir)
    
    # Generate run name if not provided
    if train_config.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_config.run_name = f"gemma3-1b-qlora-{timestamp}"
    
    # Setup TensorBoard logging directory
    if train_config.logging_dir is None:
        train_config.logging_dir = os.path.join(train_config.output_dir, "tensorboard")
    os.makedirs(train_config.logging_dir, exist_ok=True)
    
    # Log configuration
    logger.info("="*60)
    logger.info("GEMMA 3-1B QLoRA SFT FINE-TUNING")
    logger.info("="*60)
    logger.info(f"Run name: {train_config.run_name}")
    logger.info(f"Model: {model_config.model_name}")
    logger.info(f"LoRA rank: {lora_config.r}, alpha: {lora_config.lora_alpha}")
    logger.info(f"Learning rate: {train_config.learning_rate}")
    logger.info(f"Effective batch size: {train_config.per_device_train_batch_size} x {train_config.gradient_accumulation_steps} = {train_config.per_device_train_batch_size * train_config.gradient_accumulation_steps}")
    logger.info(f"Epochs: {train_config.num_train_epochs}")
    logger.info(f"NEFTune alpha: {train_config.neftune_noise_alpha}")
    logger.info(f"TensorBoard logs: {train_config.logging_dir}")
    logger.info("="*60)
    
    # Load dataset with train/eval split
    logger.info("Loading dataset...")
    train_dataset, eval_dataset = load_dataset_split(dataset_path, eval_dataset_path)
    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # Setup model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(model_config)
    
    # Apply LoRA
    logger.info("Applying LoRA adapters...")
    model, peft_config = setup_lora(model, lora_config)
    
    # Build training arguments - version aware
    training_kwargs = {
        "output_dir": train_config.output_dir,
        "run_name": train_config.run_name,
        "num_train_epochs": train_config.num_train_epochs,
        "max_steps": train_config.max_steps,
        "per_device_train_batch_size": train_config.per_device_train_batch_size,
        "per_device_eval_batch_size": train_config.per_device_eval_batch_size,
        "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
        "learning_rate": train_config.learning_rate,
        "warmup_ratio": train_config.warmup_ratio,
        "lr_scheduler_type": train_config.lr_scheduler_type,
        "weight_decay": train_config.weight_decay,
        "max_grad_norm": train_config.max_grad_norm,
        "fp16": train_config.fp16,
        "bf16": train_config.bf16,
        "gradient_checkpointing": train_config.gradient_checkpointing,
        "optim": train_config.optim,
        "logging_dir": train_config.logging_dir,
        "logging_steps": train_config.logging_steps,
        "logging_first_step": train_config.logging_first_step,
        "save_steps": train_config.save_steps,
        "save_total_limit": train_config.save_total_limit,
        "save_strategy": train_config.save_strategy,
        "eval_steps": train_config.eval_steps,
        "seed": train_config.seed,
        "report_to": train_config.report_to,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }
    
    # Add version-specific parameters
    if USE_SFT_CONFIG:
        # TRL >= 0.14 uses SFTConfig
        training_kwargs["eval_strategy"] = train_config.eval_strategy
    else:
        # Older TRL uses TrainingArguments
        training_kwargs["evaluation_strategy"] = train_config.eval_strategy
    
    training_args = SFTConfig(**training_kwargs)
    
    # Setup callbacks
    callbacks = []
    
    # Early stopping callback
    if train_config.early_stopping_patience > 0:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=train_config.early_stopping_patience,
            early_stopping_threshold=train_config.early_stopping_threshold,
        )
        callbacks.append(early_stopping)
        logger.info(f"Early stopping enabled: patience={train_config.early_stopping_patience}")
    
    # Initialize SFTTrainer - version aware
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "args": training_args,
        "peft_config": peft_config,
        "callbacks": callbacks,
    }
    
    if USE_SFT_CONFIG:
        # TRL >= 0.14 - params in SFTConfig, use processing_class
        trainer_kwargs["processing_class"] = tokenizer
    else:
        # Older TRL - params in SFTTrainer
        trainer_kwargs["tokenizer"] = tokenizer
        trainer_kwargs["max_seq_length"] = model_config.max_seq_length
        trainer_kwargs["dataset_text_field"] = "text"
        trainer_kwargs["packing"] = False
    
    trainer = SFTTrainer(**trainer_kwargs)
    
    # Start training
    logger.info("="*60)
    logger.info("Starting training...")
    logger.info("="*60)
    
    # Resume from checkpoint if specified
    resume_checkpoint = train_config.resume_from_checkpoint
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
    
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    # Log training results
    logger.info("Training completed!")
    logger.info(f"Training loss: {train_result.training_loss:.4f}")
    logger.info(f"Training steps: {train_result.global_step}")
    
    # Save final model
    final_model_path = os.path.join(train_config.output_dir, "final_model")
    logger.info(f"Saving final model to: {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Save training state (for potential resume)
    trainer.save_state()
    
    # Save training metrics
    metrics_path = os.path.join(train_config.output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(train_result.metrics, f, indent=2)
    logger.info(f"Training metrics saved to: {metrics_path}")
    
    # Save complete training config
    config_path = os.path.join(train_config.output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "run_name": train_config.run_name,
            "model_name": model_config.model_name,
            "max_seq_length": model_config.max_seq_length,
            "lora_r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "learning_rate": train_config.learning_rate,
            "epochs": train_config.num_train_epochs,
            "batch_size": train_config.per_device_train_batch_size,
            "gradient_accumulation": train_config.gradient_accumulation_steps,
            "effective_batch_size": train_config.per_device_train_batch_size * train_config.gradient_accumulation_steps,
            "warmup_ratio": train_config.warmup_ratio,
            "weight_decay": train_config.weight_decay,
            "neftune_noise_alpha": train_config.neftune_noise_alpha,
            "training_loss": train_result.training_loss,
            "total_steps": train_result.global_step,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    logger.info(f"Training config saved to: {config_path}")
    
    # Summary
    logger.info("="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    logger.info(f"Final model: {final_model_path}")
    logger.info(f"TensorBoard: tensorboard --logdir {train_config.logging_dir}")
    logger.info(f"Training loss: {train_result.training_loss:.4f}")
    logger.info("="*60)

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Model saved to: {final_model_path}")
    print(f"Config saved to: {config_path}")
    print("="*60)
    
    return trainer

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="QLoRA SFT Fine-tuning for Gemma 3-1B")
    parser.add_argument("--dataset", type=str, 
                        default="../data/final/multiturn_dataset_cleaned_no_thought.json",
                        help="Path to training dataset")
    parser.add_argument("--eval_dataset", type=str, default=None,
                        help="Path to evaluation dataset (optional, auto-split if not provided)")
    parser.add_argument("--output_dir", type=str, 
                        default="./outputs/gemma3-1b-qlora-sft",
                        help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=32,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="LoRA alpha (typically 2x rank)")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Setup configs from args
    model_config = ModelConfig(
        max_seq_length=args.max_seq_length
    )
    
    train_config = TrainConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
    )
    
    lora_config = LoRAConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    
    # Run training
    train(
        dataset_path=args.dataset,
        eval_dataset_path=args.eval_dataset,
        model_config=model_config,
        train_config=train_config,
        lora_config=lora_config,
    )

