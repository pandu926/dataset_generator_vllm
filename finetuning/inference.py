"""
Inference Script for PMB UNSIQ Fine-tuned Model
Supports both local LoRA adapters and HuggingFace models
"""

import torch
import argparse
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_pipeline(model_path: str, use_4bit: bool = True):
    """Load model using transformers pipeline (simplest method)."""
    print(f"Loading model: {model_path}")
    
    pipe = pipeline(
        "text-generation",
        model=model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    print("Model loaded!")
    return pipe, None

def load_model_with_lora(
    base_model: str, 
    adapter_path: str, 
    use_4bit: bool = True
):
    """Load base model with LoRA adapters (for local training results)."""
    print(f"Loading base model: {base_model}")
    print(f"Loading adapters from: {adapter_path}")
    
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(model, adapter_path)
    
    print("Model loaded with LoRA adapters!")
    return model, tokenizer

# =============================================================================
# GENERATION
# =============================================================================

def generate_response_pipeline(
    pipe, 
    question: str, 
    max_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.7,
) -> str:
    """Generate response using pipeline."""
    prompt = f"""<start_of_turn>user
{question}
<end_of_turn>
<start_of_turn>assistant
"""
    result = pipe(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
    )
    text = result[0]["generated_text"]
    return text.split("<start_of_turn>assistant")[-1].strip()

def generate_response_model(
    model, 
    tokenizer, 
    question: str, 
    max_tokens: int = 256,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> str:
    """Generate response using model + tokenizer."""
    prompt = f"""<start_of_turn>user
{question}
<end_of_turn>
<start_of_turn>assistant
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return response.split("<start_of_turn>assistant")[-1].split("<end_of_turn>")[0].strip()

# =============================================================================
# INTERACTIVE CHAT
# =============================================================================

def interactive_chat(pipe=None, model=None, tokenizer=None):
    """Interactive chat loop."""
    print("\n" + "="*60)
    print("PMB UNSIQ ASSISTANT - INTERACTIVE CHAT")
    print("="*60)
    print("Ketik pertanyaan Anda tentang UNSIQ.")
    print("Ketik 'quit' atau 'exit' untuk keluar.\n")
    
    while True:
        try:
            user_input = input("User: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Sampai jumpa!")
                break
            
            if not user_input:
                continue
            
            if pipe:
                response = generate_response_pipeline(pipe, user_input)
            else:
                response = generate_response_model(model, tokenizer, user_input)
            
            print(f"\nAssistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\nSampai jumpa!")
            break

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PMB UNSIQ Model Inference")
    parser.add_argument("--model_path", type=str, 
                        default="Pandusu/gemma3-pmb-unsiq-qlora-v10",
                        help="HuggingFace model or local adapter path")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model (required for local LoRA adapters)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt to process")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--no_4bit", action="store_true",
                        help="Disable 4-bit quantization")
    
    args = parser.parse_args()
    
    # Determine loading method
    if args.base_model:
        # Local LoRA adapters
        model, tokenizer = load_model_with_lora(
            args.base_model, 
            args.model_path,
            use_4bit=not args.no_4bit
        )
        pipe = None
    else:
        # HuggingFace model or merged model
        pipe, _ = load_model_pipeline(args.model_path, use_4bit=not args.no_4bit)
        model, tokenizer = None, None
    
    if args.prompt:
        # Single prompt mode
        if pipe:
            response = generate_response_pipeline(pipe, args.prompt)
        else:
            response = generate_response_model(model, tokenizer, args.prompt)
        print(f"\nQuestion: {args.prompt}")
        print(f"\nAnswer: {response}")
    else:
        # Interactive mode
        interactive_chat(pipe=pipe, model=model, tokenizer=tokenizer)

if __name__ == "__main__":
    main()
