"""
Testing Script for Fine-tuned PMB UNSIQ Models

Script untuk testing model finetuning dengan:
- 5 sample pertanyaan dari dataset test
- Model dari finetuning/model_hasil_reserach_parameter
- Max tokens: 1024
- Temperature: 0.4
- do_sample: True
"""

import json
import random
import os
import sys
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path untuk dataset test
TEST_DATA_PATH = "/workspace/dataset_generator_vllm/data/final/split/merged_all_categories_test_no_cot.json"

# Base model name (sesuai dengan adapter config)
BASE_MODEL = "google/gemma-3-1b-it"

# Path ke model hasil research parameter
MODEL_BASE_DIR = "/workspace/dataset_generator_vllm/finetuning/model_hasil_reserach_parameter"

# List model yang tersedia
AVAILABLE_MODELS = [
    "r16_a32_lr1e-4_b2_e3_ga8",
    "r16_a32_lr2e-4_b2_e3_ga8",
    "r16_a32_lr2e-4_b2_e4_ga8",
    "r32_a64_lr2e-4_b2_e3_ga8",
    "r8_a16_lr2e-4_b2_e3_ga8",
    "r8_a16_lr2e-4_b2_e5_ga8",
]

# Generation config
# do_sample=False = greedy decoding (lebih akurat dan deterministik)
# Jika do_sample=False, temperature/top_p/top_k tidak digunakan
GENERATION_CONFIG = {
    "max_new_tokens": 1024,
    "temperature": 0.4,
    "do_sample": False,  # Greedy decoding untuk akurasi lebih tinggi
    "top_p": 0.95,
    "top_k": 100,  # Increased from 50
}

# Jumlah sample pertanyaan
NUM_SAMPLES = 5

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_test_data(data_path: str) -> list:
    """Load test dataset dari file JSON."""
    print(f"📂 Loading test data from: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"   Total data: {len(data)} entries")
    return data


def extract_questions(data: list, num_samples: int = 5) -> list:
    """
    Extract user questions from conversations.
    Returns list of dict with question and expected answer.
    """
    questions = []
    
    for item in data:
        conversation = item.get("conversation", [])
        if len(conversation) >= 2:
            # Ambil pertanyaan pertama dari user dan jawaban pertama dari model
            user_msg = None
            model_response = None
            
            for i, turn in enumerate(conversation):
                if turn.get("role") == "user" and user_msg is None:
                    user_msg = turn.get("content", "")
                elif turn.get("role") == "model" and user_msg is not None and model_response is None:
                    model_response = turn.get("content", "")
                    break
            
            if user_msg and model_response:
                questions.append({
                    "id": item.get("id", "unknown"),
                    "category": item.get("category", "unknown"),
                    "question": user_msg,
                    "expected_answer": model_response,
                })
    
    # Random sample
    if len(questions) > num_samples:
        random.seed(42)  # Untuk reproducibility
        questions = random.sample(questions, num_samples)
    
    return questions


def load_model_and_tokenizer(model_name: str):
    """Load base model, LoRA adapter, dan tokenizer."""
    adapter_path = os.path.join(MODEL_BASE_DIR, model_name, "final_model")
    
    print(f"\n🔧 Loading model configuration...")
    print(f"   Base model: {BASE_MODEL}")
    print(f"   Adapter path: {adapter_path}")
    
    # Load tokenizer
    print(f"📚 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print(f"🤖 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Load adapter
    print(f"🔌 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    return model, tokenizer


def generate_response(model, tokenizer, question: str, gen_config: dict) -> str:
    """Generate response untuk pertanyaan."""
    # Format prompt sesuai format Gemma 3
    prompt = f"""<start_of_turn>user
{question}<end_of_turn>
<start_of_turn>model
"""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_config["max_new_tokens"],
            temperature=gen_config["temperature"],
            do_sample=gen_config["do_sample"],
            top_p=gen_config.get("top_p", 0.9),
            top_k=gen_config.get("top_k", 50),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract response (setelah <start_of_turn>model)
    if "<start_of_turn>model" in full_output:
        response = full_output.split("<start_of_turn>model")[-1]
        # Clean up
        response = response.replace("<end_of_turn>", "").replace("<eos>", "").strip()
    else:
        response = full_output[len(prompt):].strip()
    
    return response


def run_test(model_name: str = None):
    """Run testing untuk satu atau semua model."""
    # Load test data
    test_data = load_test_data(TEST_DATA_PATH)
    
    # Extract questions
    questions = extract_questions(test_data, NUM_SAMPLES)
    
    print(f"\n📋 Extracted {len(questions)} sample questions:")
    for i, q in enumerate(questions, 1):
        print(f"   {i}. [{q['category']}] {q['question'][:60]}...")
    
    # Determine which models to test
    if model_name:
        models_to_test = [model_name]
    else:
        models_to_test = AVAILABLE_MODELS
    
    # Results storage
    all_results = []
    
    for model_name in models_to_test:
        print(f"\n{'='*80}")
        print(f"📊 Testing Model: {model_name}")
        print(f"{'='*80}")
        
        try:
            # Load model
            model, tokenizer = load_model_and_tokenizer(model_name)
            
            # Test setiap pertanyaan
            model_results = []
            
            for i, q in enumerate(questions, 1):
                print(f"\n📝 Question {i}/{len(questions)}")
                print(f"   Category: {q['category']}")
                print(f"   ID: {q['id']}")
                print(f"\n   ❓ Question:")
                print(f"   {q['question']}")
                
                # Generate response
                print(f"\n   🔄 Generating response...")
                response = generate_response(model, tokenizer, q['question'], GENERATION_CONFIG)
                
                print(f"\n   🤖 Generated Response:")
                print(f"   {response}")
                
                print(f"\n   ✅ Expected Answer:")
                print(f"   {q['expected_answer']}")
                
                print(f"\n   {'-'*70}")
                
                model_results.append({
                    "id": q['id'],
                    "category": q['category'],
                    "question": q['question'],
                    "expected_answer": q['expected_answer'],
                    "generated_response": response,
                })
            
            all_results.append({
                "model_name": model_name,
                "timestamp": datetime.now().isoformat(),
                "generation_config": GENERATION_CONFIG,
                "results": model_results,
            })
            
            # Cleanup memory
            del model
            del tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Error testing model {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save results
    output_path = "/workspace/dataset_generator_vllm/finetuning/test_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Results saved to: {output_path}")
    
    return all_results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("🚀 PMB UNSIQ Fine-tuned Model Testing Script")
    print("="*80)
    
    print(f"\n⚙️ Generation Config:")
    print(f"   - max_new_tokens: {GENERATION_CONFIG['max_new_tokens']}")
    print(f"   - temperature: {GENERATION_CONFIG['temperature']}")
    print(f"   - do_sample: {GENERATION_CONFIG['do_sample']}")
    print(f"   - top_p: {GENERATION_CONFIG.get('top_p', 0.9)}")
    print(f"   - top_k: {GENERATION_CONFIG.get('top_k', 50)}")
    
    print(f"\n📂 Available models:")
    for i, m in enumerate(AVAILABLE_MODELS, 1):
        print(f"   {i}. {m}")
    
    # Check command line arguments
    if len(sys.argv) > 1:
        model_arg = sys.argv[1]
        if model_arg in AVAILABLE_MODELS:
            print(f"\n🎯 Testing single model: {model_arg}")
            run_test(model_arg)
        elif model_arg == "all":
            print(f"\n🎯 Testing all models...")
            run_test()
        else:
            print(f"\n❌ Invalid model: {model_arg}")
            print(f"   Available options: {', '.join(AVAILABLE_MODELS)}, all")
    else:
        # Default: test model pertama saja
        default_model = AVAILABLE_MODELS[0]
        print(f"\n🎯 Testing default model: {default_model}")
        print(f"   (Use 'python test_finetuned_models.py all' to test all models)")
        run_test(default_model)
