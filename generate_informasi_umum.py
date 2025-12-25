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
    # ==========================================================================
    # GENERAL TERMS & UNIVERSITY (25 skenario)
    # ==========================================================================
    {"id": "G01", "term": "UNSIQ", "scenario": "User tanya kepanjangan UNSIQ", "complexity": "direct"},
    {"id": "G02", "term": "PMB", "scenario": "User tanya arti PMB", "complexity": "direct"},
    {"id": "G03", "term": "UPT", "scenario": "User tanya apa itu UPT", "complexity": "direct"},
    {"id": "G04", "term": "BAN-PT", "scenario": "User tanya tentang BAN-PT", "complexity": "reasoning"},
    {"id": "G05", "term": "LLDIKTI", "scenario": "User tanya fungsi LLDIKTI", "complexity": "reasoning"},
    {"id": "G06", "term": "YPIIQ", "scenario": "User tanya apa itu YPIIQ", "complexity": "reasoning"},
    {"id": "G07", "term": "NIM", "scenario": "User tanya bedanya NIM dan NIK", "complexity": "reasoning"},
    {"id": "G08", "term": "KHS", "scenario": "User tanya apa itu KHS", "complexity": "direct"},
    {"id": "G09", "term": "SK", "scenario": "User tanya SK kelulusan itu apa", "complexity": "direct"},
    {"id": "G10", "term": "Ormawa", "scenario": "User tanya apa itu Ormawa", "complexity": "direct"},
    {"id": "G11", "term": "Rektor", "scenario": "User tanya siapa Rektor UNSIQ", "complexity": "direct"},
    {"id": "G12", "term": "Dekan", "scenario": "User tanya fungsi Dekan di fakultas", "complexity": "reasoning"},
    {"id": "G13", "term": "Kaprodi", "scenario": "User tanya apa tugas Kaprodi", "complexity": "reasoning"},
    {"id": "G14", "term": "Dosen PA", "scenario": "User tanya apa itu Dosen PA", "complexity": "direct"},
    {"id": "G15", "term": "IPK", "scenario": "User tanya cara menghitung IPK", "complexity": "reasoning"},
    {"id": "G16", "term": "IPS", "scenario": "User tanya bedanya IPS dan IPK", "complexity": "reasoning"},
    {"id": "G17", "term": "SKS", "scenario": "User tanya apa itu SKS", "complexity": "direct"},
    {"id": "G18", "term": "Semester", "scenario": "User tanya berapa lama 1 semester", "complexity": "direct"},
    {"id": "G19", "term": "Cuti Akademik", "scenario": "User tanya prosedur cuti akademik", "complexity": "reasoning"},
    {"id": "G20", "term": "DO", "scenario": "User tanya apa itu status DO", "complexity": "reasoning"},
    {"id": "G21", "term": "Wisuda", "scenario": "User tanya syarat wisuda", "complexity": "direct"},
    {"id": "G22", "term": "Yudisium", "scenario": "User tanya perbedaan yudisium dan wisuda", "complexity": "reasoning"},
    {"id": "G23", "term": "Transkip", "scenario": "User tanya cara dapat transkip nilai", "complexity": "direct"},
    {"id": "G24", "term": "Ijazah", "scenario": "User tanya kapan ijazah diberikan", "complexity": "direct"},
    {"id": "G25", "term": "Akreditasi", "scenario": "User tanya apa pentingnya akreditasi", "complexity": "reasoning"},

    # ==========================================================================
    # ACADEMIC PROGRAMS & PRODI (30 skenario)
    # ==========================================================================
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
    {"id": "A11", "term": "Keperawatan", "scenario": "User tanya prospek kerja prodi Keperawatan", "complexity": "reasoning"},
    {"id": "A12", "term": "Manajemen", "scenario": "User tanya prodi Manajemen di UNSIQ", "complexity": "direct"},
    {"id": "A13", "term": "Akuntansi", "scenario": "User tanya ada prodi Akuntansi tidak", "complexity": "direct"},
    {"id": "A14", "term": "Ekonomi Syariah", "scenario": "User tanya bedanya Ekonomi Syariah dengan Manajemen", "complexity": "reasoning"},
    {"id": "A15", "term": "Bahasa Arab", "scenario": "User tanya prodi Bahasa Arab", "complexity": "direct"},
    {"id": "A16", "term": "Bahasa Inggris", "scenario": "User tanya prodi Bahasa Inggris", "complexity": "direct"},
    {"id": "A17", "term": "Ilmu Politik", "scenario": "User tanya prodi Ilmu Politik", "complexity": "direct"},
    {"id": "A18", "term": "Gizi", "scenario": "User tanya prodi Gizi di FIKES", "complexity": "direct"},
    {"id": "A19", "term": "FASTIKOM", "scenario": "User tanya fakultas FASTIKOM", "complexity": "direct"},
    {"id": "A20", "term": "FIKES", "scenario": "User tanya kepanjangan FIKES", "complexity": "direct"},
    {"id": "A21", "term": "FEB", "scenario": "User tanya apa itu FEB", "complexity": "direct"},
    {"id": "A22", "term": "FSH", "scenario": "User tanya singkatan FSH", "complexity": "direct"},
    {"id": "A23", "term": "FITK", "scenario": "User tanya kepanjangan FITK", "complexity": "direct"},
    {"id": "A24", "term": "FKSP", "scenario": "User tanya apa itu FKSP", "complexity": "direct"},
    {"id": "A25", "term": "Pascasarjana", "scenario": "User tanya prodi S2 di UNSIQ", "complexity": "reasoning"},
    {"id": "A26", "term": "D3", "scenario": "User tanya prodi D3 apa saja", "complexity": "direct"},
    {"id": "A27", "term": "Reguler", "scenario": "User tanya bedanya kelas Reguler dan Ekstension", "complexity": "reasoning"},
    {"id": "A28", "term": "Ekstension", "scenario": "User tanya siapa yang cocok kelas Ekstension", "complexity": "reasoning"},
    {"id": "A29", "term": "Transfer", "scenario": "User tanya syarat mahasiswa transfer", "complexity": "reasoning"},
    {"id": "A30", "term": "Lintas Prodi", "scenario": "User tanya bisa tidak ambil matkul lintas prodi", "complexity": "reasoning"},

    # ==========================================================================
    # SYSTEMS & PLATFORMS (25 skenario)
    # ==========================================================================
    {"id": "S01", "term": "SIAKAD", "scenario": "User tanya fungsi SIAKAD", "complexity": "direct"},
    {"id": "S02", "term": "MBKM", "scenario": "User tanya kepanjangan MBKM", "complexity": "direct"},
    {"id": "S03", "term": "RPL", "scenario": "User tanya jalur RPL itu apa", "complexity": "reasoning"},
    {"id": "S04", "term": "LMS", "scenario": "User tanya apa itu LMS", "complexity": "direct"},
    {"id": "S05", "term": "SISTER", "scenario": "User tanya website SISTER untuk apa", "complexity": "reasoning"},
    {"id": "S06", "term": "QRIS", "scenario": "User tanya pembayaran pakai QRIS", "complexity": "direct"},
    {"id": "S07", "term": "KKNI", "scenario": "User tanya apa itu KKNI", "complexity": "reasoning"},
    {"id": "S08", "term": "SNP", "scenario": "User tanya standar SNP", "complexity": "reasoning"},
    {"id": "S09", "term": "LTPQ", "scenario": "User tanya peran LTPQ", "complexity": "reasoning"},
    {"id": "S10", "term": "E-Learning", "scenario": "User tanya platform e-learning UNSIQ", "complexity": "direct"},
    {"id": "S11", "term": "Portal Akademik", "scenario": "User tanya cara akses portal akademik", "complexity": "direct"},
    {"id": "S12", "term": "KRS", "scenario": "User tanya apa itu KRS", "complexity": "direct"},
    {"id": "S13", "term": "KRS Online", "scenario": "User tanya cara isi KRS online", "complexity": "reasoning"},
    {"id": "S14", "term": "UTS", "scenario": "User tanya jadwal UTS", "complexity": "direct"},
    {"id": "S15", "term": "UAS", "scenario": "User tanya bedanya UTS dan UAS", "complexity": "reasoning"},
    {"id": "S16", "term": "Praktikum", "scenario": "User tanya apa itu praktikum", "complexity": "direct"},
    {"id": "S17", "term": "PKL", "scenario": "User tanya syarat PKL", "complexity": "reasoning"},
    {"id": "S18", "term": "KKN", "scenario": "User tanya apa itu KKN", "complexity": "direct"},
    {"id": "S19", "term": "Skripsi", "scenario": "User tanya syarat mengambil skripsi", "complexity": "reasoning"},
    {"id": "S20", "term": "Tesis", "scenario": "User tanya bedanya skripsi dan tesis", "complexity": "reasoning"},
    {"id": "S21", "term": "Jurnal", "scenario": "User tanya wajib tidak publikasi jurnal", "complexity": "reasoning"},
    {"id": "S22", "term": "Turnitin", "scenario": "User tanya apa itu Turnitin", "complexity": "direct"},
    {"id": "S23", "term": "Perpustakaan Digital", "scenario": "User tanya cara akses perpustakaan digital", "complexity": "direct"},
    {"id": "S24", "term": "SINTA", "scenario": "User tanya apa itu SINTA", "complexity": "reasoning"},
    {"id": "S25", "term": "Google Scholar", "scenario": "User tanya UNSIQ ada profil Google Scholar tidak", "complexity": "direct"},

    # ==========================================================================
    # DOCUMENTS & ADMINISTRATION (25 skenario)
    # ==========================================================================
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
    {"id": "D11", "term": "Slip Gaji", "scenario": "User tanya dokumen slip gaji untuk apa", "complexity": "reasoning"},
    {"id": "D12", "term": "SKHUN", "scenario": "User tanya apa itu SKHUN", "complexity": "direct"},
    {"id": "D13", "term": "Rapor", "scenario": "User tanya rapor semester berapa yang diperlukan", "complexity": "reasoning"},
    {"id": "D14", "term": "Pas Foto", "scenario": "User tanya spesifikasi pas foto pendaftaran", "complexity": "direct"},
    {"id": "D15", "term": "Surat Sehat", "scenario": "User tanya surat sehat dari mana", "complexity": "reasoning"},
    {"id": "D16", "term": "Surat Rekomendasi", "scenario": "User tanya surat rekomendasi untuk apa", "complexity": "reasoning"},
    {"id": "D17", "term": "Legalisir", "scenario": "User tanya cara legalisir ijazah", "complexity": "direct"},
    {"id": "D18", "term": "Materai", "scenario": "User tanya dokumen mana yang perlu materai", "complexity": "reasoning"},
    {"id": "D19", "term": "Surat Pernyataan", "scenario": "User tanya isi surat pernyataan bermaterai", "complexity": "reasoning"},
    {"id": "D20", "term": "Formulir", "scenario": "User tanya formulir pendaftaran di mana", "complexity": "direct"},
    {"id": "D21", "term": "Kwitansi", "scenario": "User tanya cara dapat kwitansi pembayaran", "complexity": "direct"},
    {"id": "D22", "term": "Surat Tugas", "scenario": "User tanya apa itu surat tugas dosen", "complexity": "reasoning"},
    {"id": "D23", "term": "Berita Acara", "scenario": "User tanya fungsi berita acara", "complexity": "reasoning"},
    {"id": "D24", "term": "Surat Keterangan", "scenario": "User tanya cara minta surat keterangan aktif kuliah", "complexity": "direct"},
    {"id": "D25", "term": "Alumni", "scenario": "User tanya cara daftar alumni UNSIQ", "complexity": "direct"},

    # ==========================================================================
    # SCHOLARSHIPS & EXTERNAL (25 skenario)
    # ==========================================================================
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
    {"id": "E11", "term": "Bidikmisi", "scenario": "User tanya bedanya Bidikmisi dan KIP", "complexity": "reasoning"},
    {"id": "E12", "term": "Beasiswa Prestasi", "scenario": "User tanya syarat beasiswa prestasi", "complexity": "reasoning"},
    {"id": "E13", "term": "Beasiswa Hafidz", "scenario": "User tanya beasiswa untuk hafidz Quran", "complexity": "reasoning"},
    {"id": "E14", "term": "Beasiswa Yatim", "scenario": "User tanya beasiswa untuk anak yatim", "complexity": "direct"},
    {"id": "E15", "term": "Beasiswa Dhuafa", "scenario": "User tanya syarat beasiswa dhuafa", "complexity": "reasoning"},
    {"id": "E16", "term": "LPDP", "scenario": "User tanya UNSIQ bisa untuk LPDP tidak", "complexity": "reasoning"},
    {"id": "E17", "term": "Djarum", "scenario": "User tanya beasiswa Djarum", "complexity": "direct"},
    {"id": "E18", "term": "Kemenag", "scenario": "User tanya beasiswa dari Kemenag", "complexity": "direct"},
    {"id": "E19", "term": "Kemendikbud", "scenario": "User tanya bantuan dari Kemendikbud", "complexity": "reasoning"},
    {"id": "E20", "term": "BLT", "scenario": "User tanya bantuan BLT untuk mahasiswa", "complexity": "reasoning"},
    {"id": "E21", "term": "UKT", "scenario": "User tanya apa itu UKT", "complexity": "direct"},
    {"id": "E22", "term": "SPP", "scenario": "User tanya bedanya UKT dan SPP", "complexity": "reasoning"},
    {"id": "E23", "term": "Potongan", "scenario": "User tanya potongan biaya kuliah", "complexity": "reasoning"},
    {"id": "E24", "term": "Cicilan", "scenario": "User tanya sistem cicilan pembayaran", "complexity": "direct"},
    {"id": "E25", "term": "Keringanan", "scenario": "User tanya cara mengajukan keringanan biaya", "complexity": "reasoning"},

    # ==========================================================================
    # CAMPUS LIFE & FACILITIES (20 skenario)
    # ==========================================================================
    {"id": "C01", "term": "UKM", "scenario": "User tanya apa itu UKM di kampus", "complexity": "direct"},
    {"id": "C02", "term": "BEM", "scenario": "User tanya fungsi BEM", "complexity": "reasoning"},
    {"id": "C03", "term": "HIMA", "scenario": "User tanya apa itu HIMA", "complexity": "direct"},
    {"id": "C04", "term": "Laboratorium", "scenario": "User tanya fasilitas laboratorium UNSIQ", "complexity": "direct"},
    {"id": "C05", "term": "Masjid", "scenario": "User tanya masjid kampus", "complexity": "direct"},
    {"id": "C06", "term": "Perpustakaan", "scenario": "User tanya jam buka perpustakaan", "complexity": "direct"},
    {"id": "C07", "term": "Aula", "scenario": "User tanya lokasi aula kampus", "complexity": "direct"},
    {"id": "C08", "term": "Gedung Rektorat", "scenario": "User tanya di mana gedung rektorat", "complexity": "direct"},
    {"id": "C09", "term": "Kantin", "scenario": "User tanya fasilitas kantin", "complexity": "direct"},
    {"id": "C10", "term": "Hotspot", "scenario": "User tanya akses WiFi kampus", "complexity": "direct"},
    {"id": "C11", "term": "Parkir", "scenario": "User tanya area parkir mahasiswa", "complexity": "direct"},
    {"id": "C12", "term": "Asrama", "scenario": "User tanya fasilitas asrama mahasiswa", "complexity": "reasoning"},
    {"id": "C13", "term": "Klinik", "scenario": "User tanya fasilitas kesehatan kampus", "complexity": "direct"},
    {"id": "C14", "term": "ATM", "scenario": "User tanya lokasi ATM di kampus", "complexity": "direct"},
    {"id": "C15", "term": "Fotokopi", "scenario": "User tanya tempat fotokopi di kampus", "complexity": "direct"},
    {"id": "C16", "term": "Mushola", "scenario": "User tanya mushola di tiap gedung", "complexity": "direct"},
    {"id": "C17", "term": "Toilet", "scenario": "User tanya fasilitas toilet di kampus", "complexity": "direct"},
    {"id": "C18", "term": "Ruang Kelas", "scenario": "User tanya kapasitas ruang kelas", "complexity": "direct"},
    {"id": "C19", "term": "Auditorium", "scenario": "User tanya kapasitas auditorium", "complexity": "direct"},
    {"id": "C20", "term": "Lapangan", "scenario": "User tanya fasilitas olahraga", "complexity": "direct"},
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
