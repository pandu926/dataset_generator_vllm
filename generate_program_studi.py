#!/usr/bin/env python3
"""
Program Studi (Prodi) Multi-turn Dataset Generator
Generates conversations about study programs, faculties, accreditation, and career prospects.
Uses `pmb_program_studi.csv` for structural data and extracts qualitative insights from seed JSON.

Usage: python generate_program_studi.py
"""

import os
import csv
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
# DATA LOADING
# =============================================================================

def load_prodi_csv(base_path: str = "new_dokument_rag") -> List[Dict]:
    """Loads structure data from CSV"""
    file_path = os.path.join(base_path, "pmb_program_studi.csv")
    prodi_list = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prodi_list.append(row)
        return prodi_list
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return []

def load_seed_json(base_path: str = "data/seeds") -> List[Dict]:
    """Loads qualitative insights from seed JSON"""
    file_path = os.path.join(base_path, "dataset_program_pendidikan_atau_jurusan.json")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        # Fallback if seed not found, though we expect it to exist
        return []

# =============================================================================
# SCENARIO GENERATION LOGIC
# =============================================================================

def generate_scenarios(prodi_list: List[Dict], seed_data: List[Dict]) -> List[Dict]:
    """
    Generates 100 scenarios by:
    1. Creating specific scenarios for each Prodi (Akreditasi, Gelar, Prospek)
    2. Creating comparative scenarios (Bedanya A dan B)
    3. Creating faculty-level scenarios
    4. Using specific FAQs from seed data
    """
    scenarios = []
    
    # 1. Basic Info for each Prodi (approx 20 prodi * 2 scenarios = 40)
    for p in prodi_list:
        name = p['Program_Studi']
        scenarios.append({
            "id": f"P-{name.replace(' ', '')[:4]}-01",
            "type": "info_dasar",
            "prodi": name,
            "scenario": f"User tanya detail program studi {name} (akreditasi, jenjang)",
            "context_key": name 
        })
        scenarios.append({
            "id": f"P-{name.replace(' ', '')[:4]}-02",
            "type": "prospek",
            "prodi": name,
            "scenario": f"User tanya prospek kerja lulusan {name}",
            "context_key": name
        })

    # 2. Extract Interesting FAQs from Seed (Selection of ~30)
    # We filter seed items that are interesting qualatative questions
    interesting_topics = [
        "akreditasi institusi", "pembeda D3 dan S1", "mengapa akreditasi penting",
        "kuliah sambil kerja", "program pascasarjana", "beasiswa prodi",
        "laki-laki di kebidanan", "jurusan untuk IPA/IPS", "magang", 
        "laboratorium", "keunggulan lulusan"
    ]
    
    count_seed = 0
    for item in seed_data:
        q = item['Q']
        if any(topic in q.lower() for topic in interesting_topics) and count_seed < 30:
             scenarios.append({
                "id": f"S-SEED-{count_seed}",
                "type": "seed_faq",
                "prodi": "Umum",
                "scenario": f"User bertanya: {q}",
                "manual_context": item['A'] # Use the seed answer as context directly
            })
             count_seed += 1
             
    # 3. Comparative & Advisory (20 items)
    comparisons = [
        ("Teknik Informatika", "Manajemen Informatika"),
        ("Sastra Inggris", "Pendidikan Bahasa Inggris"),
        ("Hukum Ekonomi Syariah", "Perbankan Syariah"),
        ("Keperawatan D3", "Keperawatan S1"),
        ("Pendidikan Agama Islam", "Pendidikan Bahasa Arab")
    ]
    
    for p1, p2 in comparisons:
        scenarios.append({
            "id": f"C-{p1[:3]}{p2[:3]}",
            "type": "comparison",
            "prodi": f"{p1} vs {p2}",
            "scenario": f"User bingung memilih antara {p1} dan {p2}",
            "context_key": "COMPARE"
        })

    # 4. Curriculum & Student Life (Fill to 100)
    # Add ~30 more items
    for p in prodi_list:
        name = p['Program_Studi']
        scenarios.append({
            "id": f"K-{name.replace(' ', '')[:4]}",
            "type": "curriculum",
            "prodi": name,
            "scenario": f"User tanya mata kuliah khas atau kurikulum di {name}",
            "context_key": name
        })
        
    # 5. Faculty level (remaining)
    faculties = list(set([p['Fakultas'] for p in prodi_list]))
    for f in faculties:
        scenarios.append({
            "id": f"F-{f[:5]}",
            "type": "faculty",
            "prodi": f,
            "scenario": f"User tanya daftar prodi di Fakultas {f}",
            "context_key": f
        })

    # Ensure unique IDs just in case
    # Trim or pad
    return scenarios[:100]

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

SYSTEM_PROMPT = """You are an expert Academic Advisor AI for UNSIQ.
Generate HIGH-QUALITY, REALISTIC MULTI-TURN conversations about Study Programs.

STRICT RULES:
1. **CONTEXT**: Use ONLY facts from provided context. Do NOT hallucinate accreditation or degrees.
2. **STYLE**: Formal, professional, encouraging, and clear.
3. **RESPONSE**:
   - Use "Anda", not "kamu".
   - "Baik," "Tentu," "Berikut informasinya:"
   - NO casual slang ("Gitu", "Lho", "Kok").
4. **FORMAT**: Clean JSON.
"""

USER_PROMPT_TEMPLATE = """
CONTEXT DATA:
{context}

SCENARIO: {scenario}
PERSONA: {persona_name} - {persona_desc}

Generate a 3-turn conversation:
1. User asks initial question.
2. AI answers with data (Akreditasi, Gelar, dll).
3. User asks follow-up (Career, difficulty, etc).
4. AI answers persuasively but honestly based on context.

OUTPUT (JSON):
"""

# =============================================================================
# MAIN EXECUTOR
# =============================================================================

def build_context(scenario, prodi_list, seed_data):
    """Constructs context string based on scenario type"""
    
    # 1. If Manual Context from Seed
    if "manual_context" in scenario:
        return f"FACT: {scenario['manual_context']}"
    
    # 2. Specific Prodi Context
    if scenario['type'] in ['info_dasar', 'prospek']:
        p_data = next((p for p in prodi_list if p['Program_Studi'] == scenario['prodi']), None)
        if p_data:
            # Try to find extra qualitative info from seed
            extra_info = ""
            for s in seed_data:
                if scenario['prodi'] in s['Q'] or scenario['prodi'] in s['A']:
                    extra_info += f"\n- {s['A']}"
            
            return f"""
            PROGRAM STUDI: {p_data['Program_Studi']}
            JENJANG: {p_data['Jenjang']}
            FAKULTAS: {p_data['Fakultas']}
            AKREDITASI: {p_data['Akreditasi']}
            DURASI: {p_data['Durasi_Tahun']} Tahun
            
            INFORMASI TAMBAHAN:
            {extra_info[:1000]}
            """
            
    # 3. Faculty Context
    if scenario['type'] == 'faculty':
        f_prodis = [p for p in prodi_list if p['Fakultas'] == scenario['prodi']]
        details = "\n".join([f"- {p['Program_Studi']} ({p['Jenjang']}, Akreditasi: {p['Akreditasi']})" for p in f_prodis])
        return f"FAKULTAS: {scenario['prodi']}\nDAFTAR PRODI:\n{details}"

    # 4. Comparison
    if scenario['type'] == 'comparison':
        # Naive approach: dump all prodi csv data (it's small)
        csv_dump = "\n".join([str(p) for p in prodi_list])
        return f"DATA SEMUA PRODI:\n{csv_dump}\n\nGunakan data di atas untuk membandingkan opsi."

    # 5. Curriculum Type (Generic Context)
    if scenario['type'] == 'curriculum':
        p_data = next((p for p in prodi_list if p['Program_Studi'] == scenario['prodi']), None)
        return f"""
        PROGRAM STUDI: {p_data['Program_Studi']}
        FAKULTAS: {p_data['Fakultas']}
        KURIKULUM: Berbasis KKNI dan MBKM.
        FOKUS: Integrasi nilai-nilai Al-Qur'an dan Sains modern.
        KEGIATAN: Perkuliahan teori, praktikum laboratorium, magang, dan KKN.
        """

    return "Data tidak tersedia."

def main():
    print("="*60)
    print("PROGRAM STUDI GENERATOR (100 SCENARIOS)")
    print("="*60)
    
    if HAS_VLLM:
        engine = VLLMEngine()
    else:
        print("No vLLM.")
        return

    # Load Sources
    prodi_csv = load_prodi_csv()
    seed_json = load_seed_json()
    
    if not prodi_csv:
        return

    # Generate Logic
    scenarios = generate_scenarios(prodi_csv, seed_json)
    print(f"Generated {len(scenarios)} scenarios.")
    
    # Setup Generator
    generator = MultiTurnGenerator(engine)
    output_dir = "data/raw/categories"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "multiturn_program_studi.json")
    
    generated_data = []
    
    # Run Generation
    pbar = tqdm(total=len(scenarios))
    batch_size = 10
    
    for i in range(0, len(scenarios), batch_size):
        batch = scenarios[i:i+batch_size]
        prompts = []
        
        for sc in batch:
            ctx = build_context(sc, prodi_csv, seed_json)
            per = random.choice(list(PERSONAS.keys()))
            
            prompt = USER_PROMPT_TEMPLATE.format(
                context=ctx,
                scenario=sc['scenario'],
                persona_name=per,
                persona_desc=PERSONAS[per]
            )
            full_prompt = f"<bos><start_of_turn>user\n{SYSTEM_PROMPT}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            prompts.append(full_prompt)
            
        outputs = engine.generate_batch(prompts, max_tokens=1024, temperature=0.7)
        
        for j, out in enumerate(outputs):
            sc = batch[j]
            conv = generator._parse_response(out)
            if conv:
                generated_data.append({
                    "id": sc['id'],
                    "category": "program_studi",
                    "scenario": sc['scenario'],
                    "output": json.dumps(conv, ensure_ascii=False)
                })
                pbar.update(1)
        
        # Incremental Save
        with open(output_file, "w") as f:
            json.dump(generated_data, f, indent=2)
            
    pbar.close()
    
    # Clean version
    clean_data = []
    for item in generated_data:
        clean_data.append({
            "id": item['id'],
            "category": "program_studi",
            "conversation": json.loads(item['output'])
        })
        
    with open(output_file.replace(".json", "_clean.json"), "w") as f:
        json.dump(clean_data, f, indent=2)
        
    print("Done.")

if __name__ == "__main__":
    main()
