"""
Quick Test Script for Fine-tuned PMB UNSIQ Model
Using the model from HuggingFace: Pandusu/gemma3-pmb-unsiq-qlora-v10
"""

from transformers import pipeline

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "Pandusu/gemma3-pmb-unsiq-qlora-v10"

# =============================================================================
# LOAD MODEL
# =============================================================================

print(f"Loading model: {MODEL_NAME}")
pipe = pipeline(
    "text-generation",
    model=MODEL_NAME,
    device_map="auto"
)
print("Model loaded!")

# =============================================================================
# HELPER FUNCTION
# =============================================================================

def ask_pmb(question: str, max_tokens: int = 256) -> str:
    """Ask a question to the PMB UNSIQ assistant."""
    prompt = f"""<start_of_turn>user
{question}
<end_of_turn>
<start_of_turn>assistant
"""
    result = pipe(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=False
    )
    text = result[0]["generated_text"]
    return text.split("<start_of_turn>assistant")[-1].strip()

# =============================================================================
# TEST QUESTIONS
# =============================================================================

if __name__ == "__main__":
    test_questions = [
        "Apa itu Universitas Sains Al-Qur'an?",
        "Di mana lokasi kampus UNSIQ?",
        "Apa saja jalur pendaftaran mahasiswa baru?",
        "Berapa biaya kuliah di UNSIQ?",
        "Apa saja program studi yang tersedia?",
    ]
    
    print("\n" + "="*60)
    print("PMB UNSIQ ASSISTANT TEST")
    print("="*60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[Q{i}] {question}")
        print("-"*40)
        answer = ask_pmb(question)
        print(f"{answer}")
        print()
