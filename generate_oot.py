#!/usr/bin/env python3
"""
OOT (Out of Topic) Multi-turn Dataset Generator
Generates conversations where the user asks out-of-scope questions.
The AI must politely REFUSE and REDIRECT to UNSIQ PMB topics.

Usage: python generate_oot.py
"""

import os
import json
import random
from typing import List, Dict
from tqdm import tqdm

SEED = 888
random.seed(SEED)

# Import modules
from src.llm_multiturn_generator import MultiTurnGenerator, PERSONAS
try:
    from src.vllm_engine import VLLMEngine
    HAS_VLLM = True
except ImportError:
    print("Warning: vLLM not found.")
    HAS_VLLM = False

# =============================================================================
# SCENARIOS (OUT OF TOPIC)
# =============================================================================

SCENARIOS = [
    # === GENERAL KNOWLEDGE ===
    {"id": "OOT01", "topic": "general", "scenario": "User tanya resep masakan nasi goreng", "complexity": "direct"},
    {"id": "OOT02", "topic": "general", "scenario": "User tanya prediksi cuaca hari ini", "complexity": "direct"},
    {"id": "OOT03", "topic": "general", "scenario": "User tanya siapa presiden Indonesia sekarang", "complexity": "direct"},
    {"id": "OOT04", "topic": "general", "scenario": "User tanya cara memperbaiki laptop rusak", "complexity": "reasoning"},
    {"id": "OOT05", "topic": "general", "scenario": "User curhat masalah percintaan (putus cinta)", "complexity": "reasoning"},
    
    # === COMPETITORS ===
    {"id": "OOT06", "topic": "competitor", "scenario": "User tanya pendaftaran di kampus UGM/UI", "complexity": "direct"},
    {"id": "OOT07", "topic": "competitor", "scenario": "User minta perbandingan UNSIQ dengan kampus lain di Jawa Tengah", "complexity": "reasoning"},
    {"id": "OOT08", "topic": "competitor", "scenario": "User tanya biaya kuliah di kampus tetangga", "complexity": "direct"},
    
    # === TECHNICAL / MATH ===
    {"id": "OOT09", "topic": "technical", "scenario": "User minta buatkan codingan Python", "complexity": "direct"},
    {"id": "OOT10", "topic": "technical", "scenario": "User tanya soal matematika kalkulus", "complexity": "reasoning"},
    {"id": "OOT11", "topic": "technical", "scenario": "User tanya cara hack website", "complexity": "edge_case"},
    
    # === SENSITIVE / TOXIC ===
    {"id": "OOT12", "topic": "sensitive", "scenario": "User marah-marah dan berkata kasar (toxic)", "complexity": "edge_case"},
    {"id": "OOT13", "topic": "sensitive", "scenario": "User tanya pendapat politik sensitif", "complexity": "edge_case"},
    {"id": "OOT14", "topic": "sensitive", "scenario": "User menggoda AI (flirting)", "complexity": "edge_case"},
    
    # === OFF-SCOPE UNSIQ ===
    {"id": "OOT15", "topic": "off_scope", "scenario": "User tanya nilai mata kuliah mahasiswa lama (bukan PMB)", "complexity": "direct"},
    {"id": "OOT16", "topic": "off_scope", "scenario": "User tanya slip gaji dosen UNSIQ", "complexity": "edge_case"},
    {"id": "OOT17", "topic": "off_scope", "scenario": "User tanya menu kantin hari ini", "complexity": "direct"},
    {"id": "OOT18", "topic": "off_scope", "scenario": "User tanya jadwal bioskop di Wonosobo", "complexity": "direct"},
    
    # === RANDOM ===
    {"id": "OOT19", "topic": "random", "scenario": "User kirim pesan acak tidak jelas (spam)", "complexity": "direct"},
    {"id": "OOT20", "topic": "random", "scenario": "User tanya rekomendasi tempat wisata di Dieng", "complexity": "reasoning"},
]

print(f"Total scenarios: {len(SCENARIOS)}")

# =============================================================================
# PROMPT TEMPLATES - REFUSAL STYLE
# =============================================================================

SYSTEM_PROMPT = """You are an expert Synthetic Data Generator for UNSIQ PMB Bot.
Generate conversations where the User asks OUT-OF-TOPIC questions, and the AI REFUSES politely.

STRICT RULES:
1. **AI IDENTITY**: You are a Customer Service Bot for PMB UNSIQ (Penerimaan Mahasiswa Baru).
2. **SCOPE**: You ONLY answer questions about UNSIQ Registration, Profile, Costs, and Prodi.
3. **REFUSAL STRATEGY**:
   - Politely apologize ("Mohon maaf,").
   - State reasoning ("Saya hanya dapat membantu informasi terkait PMB UNSIQ.").
   - Redirect ("Apakah ada yang ingin Anda tanyakan seputar pendaftaran mahasiswa baru?").
   - NEVER answer the out-of-topic question (e.g. dont give the recipe/weather/math answer).
4. **STYLE**: Formal, professional, concise.
5. **FORMAT**: Valid JSON list.

EXAMPLE:
User: "Cara bikin nasi goreng gimana?"
AI: "Mohon maaf, saya adalah asisten virtual PMB UNSIQ dan tidak menyediakan informasi resep masakan. Apakah Anda memiliki pertanyaan terkait pendaftaran mahasiswa baru di UNSIQ?"
"""

USER_PROMPT_TEMPLATE = """
SCENARIO: {scenario}
PERSONA: {persona_name} - {persona_desc}

Generate a 2-3 turn conversation.
Turn 1: User asks the OOT question. AI refuses politely.
Turn 2: User tries to push or change topic slightly. AI maintains refusal/redirection.

OUTPUT (JSON only):
[
  {{"role": "user", "content": "..."}},
  {{"role": "model", "thought": "1. Analyze: User asks OOT. 2. Refuse: State scope. 3. Redirect: Ask about PMB.", "content": "..."}},
  ...
]
"""

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("OOT DATASET GENERATOR")
    print(f"Target: {len(SCENARIOS)} conversations")
    print("="*60)
    
    # Initialize engine
    engine = None
    if HAS_VLLM:
        engine = VLLMEngine()
        print("vLLM Engine Ready.")
    else:
        print("No vLLM engine.")
        return
    
    # Initialize generator
    generator = MultiTurnGenerator(engine)
    
    # Prepare output
    output_dir = "data/raw/categories"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "multiturn_oot.json")
    
    # Generate
    generated_data = []
    batch_size = 10
    personas_list = list(PERSONAS.keys())
    total_scenarios = len(SCENARIOS)
    
    pbar = tqdm(total=total_scenarios, desc="Generating conversations", unit="conv")
    
    for batch_start in range(0, total_scenarios, batch_size):
        batch_scenarios = SCENARIOS[batch_start:batch_start+batch_size]
        
        # Build prompts
        prompts = []
        for scenario in batch_scenarios:
            persona_key = random.choice(personas_list)
            persona_desc = PERSONAS[persona_key]
            
            prompt = USER_PROMPT_TEMPLATE.format(
                scenario=scenario["scenario"],
                persona_name=persona_key,
                persona_desc=persona_desc
            )
            
            formatted = f"<bos><start_of_turn>user\n{SYSTEM_PROMPT}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            prompts.append(formatted)
        
        # Generate batch
        outputs = engine.generate_batch(prompts, max_tokens=1024, temperature=0.7)
        
        # Parse results
        for i, response in enumerate(outputs):
            scenario = batch_scenarios[i]
            conversation = generator._parse_response(response)
            
            if conversation:
                item = {
                    "id": scenario["id"],
                    "instruction": f"Out-of-topic conversation",
                    "input": "",
                    "output": json.dumps(conversation, ensure_ascii=False),
                    "text": "",
                    "category": "out_of_topic",
                    "stage": scenario["topic"],
                    "scenario": scenario["scenario"],
                    "complexity": scenario["complexity"],
                    "source": "synthetic_oot_v1"
                }
                generated_data.append(item)
                pbar.update(1)
        
        # Checkpoint
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(generated_data, f, ensure_ascii=False, indent=2)
    
    pbar.close()
    
    # Final save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nDONE! Generated {len(generated_data)} conversations")
    print(f"Saved to: {output_file}")
    
    # Clean version
    clean_data = []
    for item in generated_data:
        clean_item = {
            'id': item.get('id'),
            'category': item.get('category'),
            'stage': item.get('stage'),
            'scenario': item.get('scenario'),
            'complexity': item.get('complexity'),
            'conversation': json.loads(item.get('output', '[]'))
        }
        clean_data.append(clean_item)
    
    clean_file = os.path.join(output_dir, "multiturn_oot_clean.json")
    with open(clean_file, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)
    
    print(f"Clean version saved to: {clean_file}")


if __name__ == "__main__":
    main()
