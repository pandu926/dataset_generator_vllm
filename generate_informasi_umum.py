#!/usr/bin/env python3
"""
Informasi Umum & Singkatan Multi-turn Dataset Generator
Generates conversations about general terms, abbreviations, and acronyms used at UNSIQ.
Uses `data_singkatan.json` as the Knowledge Base.

Usage: python generate_informasi_umum.py
"""

import os
import json
import random
from typing import List, Dict
from tqdm import tqdm

SEED = 456
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
# DATA LOADING
# =============================================================================

def load_data_singkatan(base_path: str = "new_dokument_rag") -> Dict[str, str]:
    """Loads terms and definitions from data_singkatan.json"""
    file_path = os.path.join(base_path, "data_singkatan.json")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Create a dictionary for easy retrieval: Term -> Definition
        knowledge_base = {}
        for item in data:
            # Extract term from Q (e.g., "Apa kepanjangan dari UNSIQ?" -> "UNSIQ")
            question = item["Q"]
            answer = item["A"]
            
            # Simple heuristic to extract the main term from the question
            # Q usually format: "Apa kepanjangan dari X?" or "Apa arti X?"
            key = question.replace("Apa kepanjangan dari ", "").replace("Apa arti ", "").replace("?", "").strip()
            # Clean up extra words if any (like "dalam konteks UNSIQ")
            if " dalam " in key:
                key = key.split(" dalam ")[0]
            
            knowledge_base[key] = answer
            
        print(f"Loaded {len(knowledge_base)} terms from {file_path}")
        return knowledge_base
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return {}

# =============================================================================
# SCENARIOS (50 ITEMS)
# =============================================================================

# All scenarios will use the term as the key to look up context
SCENARIOS = [
    # === GENERAL TERMS (10) ===
    {"id": "G01", "term": "UNSIQ", "scenario": "User tanya kepanjangan UNSIQ", "complexity": "direct"},
    {"id": "G02", "term": "PMB", "scenario": "User tanya arti PMB", "complexity": "direct"},
    {"id": "G03", "term": "UPT", "scenario": "User tanya apa itu UPT", "complexity": "direct"},
    {"id": "G04", "term": "BAN-PT", "scenario": "User tanya tentang BAN-PT", "complexity": "reasoning"},
    {"id": "G05", "term": "LLDIKTI", "scenario": "User tanya fungsi LLDIKTI", "complexity": "reasoning"},
    {"id": "G06", "term": "YPIIQ", "scenario": "User tanya apa itu YPIIQ", "complexity": "reasoning"},
    {"id": "G07", "term": "NIM", "scenario": "User tanya bedanya NIM dan NIK", "complexity": "reasoning"}, # Context should include both
    {"id": "G08", "term": "KHS", "scenario": "User tanya apa itu KHS", "complexity": "direct"},
    {"id": "G09", "term": "SK", "scenario": "User tanya SK kelulusan itu apa", "complexity": "direct"},
    {"id": "G10", "term": "Ormawa", "scenario": "User tanya apa itu Ormawa", "complexity": "direct"},

    # === ACADEMIC ABBREVIATIONS (10) ===
    {"id": "A01", "term": "PAI", "scenario": "User tanya kepanjangan prodi PAI", "complexity": "direct"},
    {"id": "A02", "term": "KPI", "scenario": "User tanya apa itu jurusan KPI", "complexity": "direct"},
    {"id": "A03", "term": "PIAUD", "scenario": "User tanya tentang prodi PIAUD", "complexity": "direct"},
    {"id": "A04", "term": "PGMI", "scenario": "User tanya kepanjangan PGMI", "complexity": "direct"},
    {"id": "A05", "term": "HES", "scenario": "User tanya apa itu HES (Mu'amalah)", "complexity": "reasoning"},
    {"id": "A06", "term": "AHK", "scenario": "User tanya singkatan AHK", "complexity": "direct"},
    {"id": "A07", "term": "TI", "scenario": "User tanya TI di UNSIQ itu apa", "complexity": "direct"},
    {"id": "A08", "term": "Tafsir", "scenario": "User tanya tentang prodi Tafsir", "complexity": "reasoning"},
    {"id": "A09", "term": "Kebidanan", "scenario": "User tanya prodi Kebidanan", "complexity": "direct"},
    {"id": "A10", "term": "MI", "scenario": "User tanya arti MI dalam PGMI", "complexity": "reasoning"},

    # === TIERS & SYSTEMS (10) ===
    {"id": "S01", "term": "D3", "scenario": "User tanya apa maksud jenjang D3", "complexity": "direct"},
    {"id": "S02", "term": "S1", "scenario": "User tanya bedanya S1 dan S2", "complexity": "reasoning"}, # Need S1 and S2 context
    {"id": "S03", "term": "MBKM", "scenario": "User tanya kepanjangan MBKM", "complexity": "direct"},
    {"id": "S04", "term": "RPL", "scenario": "User tanya jalur RPL itu apa", "complexity": "reasoning"},
    {"id": "S05", "term": "LMS", "scenario": "User tanya apa itu LMS", "complexity": "direct"},
    {"id": "S06", "term": "SISTER", "scenario": "User tanya website SISTER untuk apa", "complexity": "reasoning"},
    {"id": "S07", "term": "QRIS", "scenario": "User tanya pembayaran pakai QRIS", "complexity": "direct"},
    {"id": "S08", "term": "KKNI", "scenario": "User tanya apa itu KKNI", "complexity": "reasoning"},
    {"id": "S09", "term": "SNP", "scenario": "User tanya standar SNP", "complexity": "reasoning"},
    {"id": "S10", "term": "LTPQ", "scenario": "User tanya peran LTPQ", "complexity": "reasoning"},

    # === DOCUMENTS & ADMIN (10) ===
    {"id": "D01", "term": "SKTM", "scenario": "User tanya fungsi SKTM", "complexity": "direct"},
    {"id": "D02", "term": "NIK", "scenario": "User tanya kenapa perlu NIK", "complexity": "reasoning"},
    {"id": "D03", "term": "NISN", "scenario": "User tanya apa itu NISN", "complexity": "direct"},
    {"id": "D04", "term": "NPSN", "scenario": "User tanya kepanjangan NPSN", "complexity": "direct"},
    {"id": "D05", "term": "KK", "scenario": "User tanya dokumen KK itu apa", "complexity": "direct"},
    {"id": "D06", "term": "SKL", "scenario": "User tanya bedanya SKL dan Ijazah", "complexity": "reasoning"},
    {"id": "D07", "term": "STR", "scenario": "User tanya apa itu STR untuk nakes", "complexity": "reasoning"},
    {"id": "D08", "term": "UKN", "scenario": "User tanya singkatan UKN", "complexity": "direct"},
    {"id": "D09", "term": "MoU", "scenario": "User tanya apa itu MoU/MoA", "complexity": "reasoning"},
    {"id": "D10", "term": "KIP-Kuliah", "scenario": "User tanya syarat KIP-Kuliah dan kepanjangannya", "complexity": "reasoning"},

    # === SCHOLARSHIPS & EXTERNAL (10) ===
    {"id": "E01", "term": "SNBP", "scenario": "User tanya SNBP itu jalur apa", "complexity": "direct"},
    {"id": "E02", "term": "SNBT", "scenario": "User tanya kepanjangan SNBT", "complexity": "direct"},
    {"id": "E03", "term": "DTKS", "scenario": "User tanya apa hubungan DTKS dan KIP", "complexity": "reasoning"},
    {"id": "E04", "term": "PBSB", "scenario": "User tanya beasiswa PBSB", "complexity": "direct"},
    {"id": "E05", "term": "CPNS", "scenario": "User tanya apakah lulusan UNSIQ bisa daftar CPNS", "complexity": "reasoning"},
    {"id": "E06", "term": "P3K", "scenario": "User tanya apa itu P3K", "complexity": "direct"},
    {"id": "E07", "term": "BSI", "scenario": "User tanya beasiswa BSI", "complexity": "direct"},
    {"id": "E08", "term": "BAZNAS", "scenario": "User tanya bantuan BAZNAS", "complexity": "direct"},
    {"id": "E09", "term": "PKH", "scenario": "User tanya beasiswa untuk keluarga PKH", "complexity": "reasoning"},
    {"id": "E10", "term": "PPL", "scenario": "User tanya apa itu PPL untuk mahasiswa", "complexity": "direct"},
]

print(f"Total scenarios: {len(SCENARIOS)}")

# =============================================================================
# PROMPT TEMPLATES - FORMAL STYLE
# =============================================================================

SYSTEM_PROMPT = """You are an expert Synthetic Data Generator for UNSIQ (Universitas Sains Al-Qur'an).
Generate HIGH-QUALITY, REALISTIC MULTI-TURN conversations related to Glossary, Terms, and Abbreviations.

STRICT RULES:
1. **CONTEXT**: Use ONLY facts from the provided context (Definition). Do NOT hallucinate.
2. **USER STYLE**: Can be casual OR formal based on persona.
3. **AI RESPONSE STYLE**:
   - Professional, formal, helpful
   - Natural formal Indonesian: "Baik,", "Tentu,", "Berikut penjelasannya."
   - CONCISE and CLEAR.
   - Use "Anda" not "kamu".
   - If explaining an abbreviation, spell it out first, then explain.
4. **THOUGHT**: Include reasoning steps.
5. **TURNS**: Generate 2-3 turns. (User asks term -> AI defines -> User asks detail/implication -> AI clarifies).
6. **FORMAT**: Valid JSON list only.
"""

USER_PROMPT_TEMPLATE = """
TERM: {term}
DEFINITION (CONTEXT):
{context}

SCENARIO: {scenario}
PERSONA: {persona_name} - {persona_desc}

Generate a 2-3 turn conversation.
Turn 1: User asks about the term. AI explains clearly based on definition.
Turn 2: User asks follow-up (e.g., function, requirement, or difference). AI answers based on context.

OUTPUT (JSON only):
[
  {{"role": "user", "content": "..."}},
  {{"role": "model", "thought": "...", "content": "..."}},
  ...
]
"""

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("INFORMASI UMUM GENERATOR")
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
    
    # Check knowledge base
    kb = load_data_singkatan()
    if not kb:
        print("Failed to load knowledge base. Exiting.")
        return
        
    # Initialize generator
    generator = MultiTurnGenerator(engine)
    
    # Prepare output
    output_dir = "data/raw/categories"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "multiturn_informasi_umum.json")
    
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
            
            # Retrieve context for the specific term
            term = scenario["term"]
            context = kb.get(term, "Maaf, informasi tidak ditemukan.")
            
            # Special handling for comparison scenarios (e.g. S1 vs S2, NIM vs NIK)
            # If the scenario implies comparison, try to fetch the other term too if implied
            if "NIM" in term and "NIK" in scenario["scenario"]:
                 context = f"NIM: {kb.get('NIM', '-')}\n\nNIK: {kb.get('NIK', '-')}"
            elif "S1" in term and "S2" in scenario["scenario"]:
                 context = f"S1 dan S2: {kb.get('S1 dan S2', kb.get('S1', '-') + ' ' + kb.get('S2', '-'))}"
            
            prompt = USER_PROMPT_TEMPLATE.format(
                term=term,
                context=context,
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
                    "instruction": f"Explain term/abbreviation: {scenario['term']}",
                    "input": "",
                    "output": json.dumps(conversation, ensure_ascii=False),
                    "text": "",
                    "category": "informasi_umum",
                    "stage": "glossary",
                    "scenario": scenario["scenario"],
                    "complexity": scenario["complexity"],
                    "source": "synthetic_singkatan_v1"
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
    
    clean_file = os.path.join(output_dir, "multiturn_informasi_umum_clean.json")
    with open(clean_file, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)
    
    print(f"Clean version saved to: {clean_file}")


if __name__ == "__main__":
    main()
