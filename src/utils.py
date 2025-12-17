"""
Utility Functions for PMB UNSIQ Dataset Generation Pipeline
"""

import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple, Generator
from datetime import datetime
import hashlib


# =============================================================================
# Token Counting
# =============================================================================

_tokenizer = None

def get_tokenizer(model_name: str = "google/gemma-2-2b"):
    """
    Get or initialize the tokenizer (cached for performance).
    
    Args:
        model_name: Model name to load tokenizer from
        
    Returns:
        Tokenizer instance
    """
    global _tokenizer
    if _tokenizer is None:
        try:
            from transformers import AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"Warning: Could not load tokenizer {model_name}: {e}")
            print("Using simple word-based tokenization as fallback")
            return None
    return _tokenizer


def count_tokens(text: str, model_name: str = "google/gemma-2-2b") -> int:
    """
    Count tokens in text using the model's tokenizer.
    Falls back to word-based estimation if tokenizer unavailable.
    
    Args:
        text: Text to count tokens for
        model_name: Model name for tokenizer
        
    Returns:
        Number of tokens
    """
    tokenizer = get_tokenizer(model_name)
    if tokenizer:
        return len(tokenizer.encode(text, add_special_tokens=False))
    else:
        # Fallback: rough estimation (1 word ≈ 1.3 tokens)
        words = len(text.split())
        return int(words * 1.3)


def count_tokens_simple(text: str) -> int:
    """
    Simple token counting without external dependencies.
    Uses word-based estimation (1 word ≈ 1.3 tokens for Indonesian).
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Estimated number of tokens
    """
    if not text:
        return 0
    words = len(text.split())
    return int(words * 1.3)


def count_pair_tokens(instruction: str, output: str, 
                      model_name: str = "google/gemma-2-2b") -> Dict[str, int]:
    """
    Count tokens for an instruction-output pair.
    
    Args:
        instruction: Instruction/question text
        output: Output/answer text
        model_name: Model name for tokenizer
        
    Returns:
        Dictionary with token counts
    """
    instruction_tokens = count_tokens(instruction, model_name)
    output_tokens = count_tokens(output, model_name)
    
    return {
        "instruction": instruction_tokens,
        "output": output_tokens,
        "total": instruction_tokens + output_tokens
    }


def validate_token_lengths(pair: Dict[str, str], config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate token lengths against configuration constraints.
    
    Args:
        pair: Dictionary with 'instruction' and 'output' keys
        config: Configuration with min/max token settings
        
    Returns:
        Tuple of (is_valid, rejection_reason)
    """
    tokens = count_pair_tokens(pair["instruction"], pair["output"])
    
    min_inst = config.get("min_instruction_tokens", 30)
    max_inst = config.get("max_instruction_tokens", 200)
    min_out = config.get("min_output_tokens", 50)
    max_out = config.get("max_output_tokens", 250)
    max_total = config.get("max_total_tokens", 512)
    
    if tokens["instruction"] < min_inst:
        return False, f"instruction_too_short ({tokens['instruction']} < {min_inst})"
    
    if tokens["instruction"] > max_inst:
        return False, f"instruction_too_long ({tokens['instruction']} > {max_inst})"
    
    if tokens["output"] < min_out:
        return False, f"output_too_short ({tokens['output']} < {min_out})"
    
    if tokens["output"] > max_out:
        return False, f"output_too_long ({tokens['output']} > {max_out})"
    
    if tokens["total"] > max_total:
        return False, f"total_too_long ({tokens['total']} > {max_total})"
    
    return True, "valid"


# =============================================================================
# Data I/O
# =============================================================================

def save_jsonl(data: List[Dict], filepath: str, mode: str = 'w'):
    """
    Save list of dictionaries as JSONL file.
    
    Args:
        data: List of dictionaries to save
        filepath: Output file path
        mode: Write mode ('w' for overwrite, 'a' for append)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, mode, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_jsonl(filepath: str) -> List[Dict]:
    """
    Load JSONL file as list of dictionaries.
    
    Args:
        filepath: Input file path
        
    Returns:
        List of dictionaries
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_json(data: Any, filepath: str):
    """Save data as JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Any:
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_yaml(filepath: str) -> Dict:
    """Load YAML configuration file"""
    import yaml
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# =============================================================================
# Batch Processing
# =============================================================================

def batch_iterator(data: List, batch_size: int) -> Generator[List, None, None]:
    """
    Iterate through data in batches.
    
    Args:
        data: List of items
        batch_size: Size of each batch
        
    Yields:
        Batches of items
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def calculate_num_batches(total_items: int, batch_size: int) -> int:
    """Calculate number of batches needed"""
    return (total_items + batch_size - 1) // batch_size


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(data: List[Dict], filepath: str, 
                    metadata: Optional[Dict] = None):
    """
    Save checkpoint with metadata.
    
    Args:
        data: Data to save
        filepath: Checkpoint file path
        metadata: Optional metadata to include
    """
    checkpoint = {
        "timestamp": datetime.now().isoformat(),
        "total_pairs": len(data),
        "data": data
    }
    if metadata:
        checkpoint["metadata"] = metadata
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)


def load_checkpoint(filepath: str) -> Tuple[List[Dict], Dict]:
    """
    Load checkpoint file.
    
    Args:
        filepath: Checkpoint file path
        
    Returns:
        Tuple of (data, metadata)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        checkpoint = json.load(f)
    
    return checkpoint.get("data", []), checkpoint.get("metadata", {})


# =============================================================================
# Text Processing
# =============================================================================

def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    return text


def format_rupiah(amount: int) -> str:
    """Format number as Indonesian Rupiah"""
    return f"Rp. {amount:,.0f}".replace(",", ".")


def extract_numbers(text: str) -> List[int]:
    """Extract all numbers from text"""
    return [int(n.replace(".", "").replace(",", "")) 
            for n in re.findall(r'[\d.,]+', text)]


def truncate_text(text: str, max_tokens: int, 
                  model_name: str = "google/gemma-2-2b") -> str:
    """
    Truncate text to maximum token count.
    
    Args:
        text: Input text
        max_tokens: Maximum tokens
        model_name: Model for tokenizer
        
    Returns:
        Truncated text
    """
    tokenizer = get_tokenizer(model_name)
    if tokenizer:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = tokenizer.decode(tokens)
    else:
        # Fallback: word-based truncation
        words = text.split()
        max_words = int(max_tokens / 1.3)
        if len(words) > max_words:
            text = ' '.join(words[:max_words])
    
    return text


# =============================================================================
# Hashing & Deduplication
# =============================================================================

def generate_hash(text: str) -> str:
    """Generate MD5 hash for text"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def generate_pair_id(instruction: str, output: str) -> str:
    """Generate unique ID for an instruction-output pair"""
    combined = f"{instruction}||{output}"
    return generate_hash(combined)[:12]


# =============================================================================
# Statistics
# =============================================================================

def calculate_stats(data: List[Dict], 
                    instruction_key: str = "instruction",
                    output_key: str = "output") -> Dict[str, Any]:
    """
    Calculate statistics for a dataset.
    
    Args:
        data: List of pairs
        instruction_key: Key for instruction field
        output_key: Key for output field
        
    Returns:
        Dictionary with statistics
    """
    if not data:
        return {"error": "Empty dataset"}
    
    inst_tokens = []
    out_tokens = []
    
    for pair in data:
        inst = pair.get(instruction_key, "")
        out = pair.get(output_key, "")
        inst_tokens.append(count_tokens(inst))
        out_tokens.append(count_tokens(out))
    
    total_tokens = [i + o for i, o in zip(inst_tokens, out_tokens)]
    
    return {
        "total_pairs": len(data),
        "instruction_tokens": {
            "min": min(inst_tokens),
            "max": max(inst_tokens),
            "avg": sum(inst_tokens) / len(inst_tokens),
            "median": sorted(inst_tokens)[len(inst_tokens) // 2]
        },
        "output_tokens": {
            "min": min(out_tokens),
            "max": max(out_tokens),
            "avg": sum(out_tokens) / len(out_tokens),
            "median": sorted(out_tokens)[len(out_tokens) // 2]
        },
        "total_tokens": {
            "min": min(total_tokens),
            "max": max(total_tokens),
            "avg": sum(total_tokens) / len(total_tokens),
            "median": sorted(total_tokens)[len(total_tokens) // 2]
        }
    }


def print_stats(stats: Dict[str, Any], title: str = "Dataset Statistics"):
    """Print statistics in a formatted way"""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")
    print(f" Total Pairs: {stats['total_pairs']}")
    print(f"\n Instruction Tokens:")
    print(f"   Min: {stats['instruction_tokens']['min']}")
    print(f"   Max: {stats['instruction_tokens']['max']}")
    print(f"   Avg: {stats['instruction_tokens']['avg']:.1f}")
    print(f"\n Output Tokens:")
    print(f"   Min: {stats['output_tokens']['min']}")
    print(f"   Max: {stats['output_tokens']['max']}")
    print(f"   Avg: {stats['output_tokens']['avg']:.1f}")
    print(f"\n Total Tokens:")
    print(f"   Min: {stats['total_tokens']['min']}")
    print(f"   Max: {stats['total_tokens']['max']}")
    print(f"   Avg: {stats['total_tokens']['avg']:.1f}")
    print(f"{'='*50}\n")


# =============================================================================
# Validation Helpers
# =============================================================================

def validate_pair_structure(pair: Dict) -> Tuple[bool, str]:
    """
    Validate that a pair has the required structure.
    
    Args:
        pair: Dictionary to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(pair, dict):
        return False, "not_a_dict"
    
    if "instruction" not in pair:
        return False, "missing_instruction"
    
    if "output" not in pair:
        return False, "missing_output"
    
    if not pair["instruction"] or not isinstance(pair["instruction"], str):
        return False, "invalid_instruction"
    
    if not pair["output"] or not isinstance(pair["output"], str):
        return False, "invalid_output"
    
    return True, "valid"


def is_indonesian(text: str) -> bool:
    """
    Check if text appears to be in Indonesian.
    Simple heuristic based on common Indonesian words.
    """
    indonesian_words = {
        'adalah', 'dan', 'atau', 'yang', 'untuk', 'dari', 'dengan', 'ini',
        'itu', 'pada', 'ke', 'di', 'tidak', 'akan', 'bisa', 'dapat', 
        'juga', 'ada', 'apa', 'bagaimana', 'berapa', 'apakah', 'saya',
        'kami', 'anda', 'mahasiswa', 'biaya', 'pendaftaran', 'kuliah'
    }
    
    words = set(text.lower().split())
    matches = words.intersection(indonesian_words)
    
    return len(matches) >= 2
