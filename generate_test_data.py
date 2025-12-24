#!/usr/bin/env python3
"""
Generate additional TEST data for categories with low sample counts.
This script generates NEW test data without affecting training data.

Categories to generate:
- alur_pendaftaran: need 7 more (8 → 15)
- informasi_umum: need 8 more (7 → 15)
- profil_unsiq: need 14 more (1 → 15)
- out_of_topic: need 3 more (12 → 15)

Usage: python generate_test_data.py
"""

import os
import json
import random
from typing import List, Dict
from tqdm import tqdm

SEED = 999
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
# CONTEXT PER CATEGORY
# =============================================================================

CONTEXTS = {
    "alur_pendaftaran": """
## ALUR PENDAFTARAN PMB UNSIQ 2025

### TAHAP 1: AKSES WEBSITE
- Website: pmb.unsiq.ac.id
- Menu: Formulir Pendaftaran, Login, Pengumuman, FAQ

### TAHAP 2: ISI FORMULIR (10-15 menit)
- Pilih Jenjang: D3 (3 tahun), S1 (4 tahun), S2 (2-3 tahun)
- Pilih Kelas: Reguler atau Ekstension
- Data: Nama, Email, No HP, Alamat
- PENTING: Screenshot Nomor Pendaftaran & PIN!

### TAHAP 3: BAYAR BIAYA PENDAFTARAN
| Gelombang | D3 & S1 | S2 |
|-----------|---------|-----|
| Gelombang 1 | GRATIS | GRATIS |
| Gelombang 2 | Rp 250.000 | Rp 300.000 |
| Gelombang 3 | Rp 250.000 | Rp 300.000 |

Metode: Bank Jateng, Finpay

### TAHAP 4: UPLOAD DOKUMEN
- Pas foto 3x4 (background merah)
- Scan ijazah/SKL/raport
- KTP/Akte/Kartu Keluarga (pilih salah satu)

### TAHAP 5: HASIL SELEKSI
- Cek di menu "Pengumuman"
- Notifikasi via WhatsApp/Email

### TAHAP 6: DAFTAR ULANG
- Setelah dinyatakan LULUS
- Bayar UKT, Registrasi Ulang

### KONTAK PMB:
📞 (0286) 321873 | 📱 0857 7504 7504 | 📧 humas@unsiq.ac.id
""",

    "informasi_umum": """
## INFORMASI UMUM UNSIQ

### JADWAL PMB 2025
| Gelombang | Periode | Biaya D3/S1 | Biaya S2 |
|-----------|---------|-------------|----------|
| Gelombang 1 | Nov-Des 2024 | GRATIS | GRATIS |
| Gelombang 2 | Jan-Mar 2025 | Rp 250.000 | Rp 300.000 |
| Gelombang 3 | Apr-Jun 2025 | Rp 250.000 | Rp 300.000 |

### JAM LAYANAN PMB
- Senin-Jumat: 08:00-16:00 WIB
- Sabtu: 08:00-12:00 WIB
- Minggu & Tanggal Merah: TUTUP

### KONTAK RESMI
📞 (0286) 321873 (jam kerja)
📱 WhatsApp: 0857 7504 7504
📧 Email: humas@unsiq.ac.id
🌐 Website: pmb.unsiq.ac.id

### LOKASI
Kampus Utama: Jl. Raya Kalibening Km. 03, Wonosobo, Jawa Tengah

### AKREDITASI
Akreditasi Institusi: B (BAN-PT)
""",

    "profil_unsiq": """
## PROFIL UNSIQ

### IDENTITAS
- Nama: Universitas Sains Al-Qur'an (UNSIQ)
- Lokasi: Jl. Raya Kalibening Km. 03, Wonosobo, Jawa Tengah
- Didirikan: 1988
- Akreditasi: B (BAN-PT)

### VISI
Menjadi universitas unggul berbasis nilai-nilai Al-Qur'an yang menghasilkan insan berkarakter, kompeten, dan berdaya saing global.

### MISI
1. Menyelenggarakan pendidikan tinggi berbasis Al-Qur'an
2. Mengembangkan penelitian dan pengabdian masyarakat
3. Membentuk karakter Islami pada civitas akademika
4. Menjalin kerjasama nasional dan internasional

### FAKULTAS
1. Fakultas Ilmu Tarbiyah dan Keguruan
2. Fakultas Syariah dan Hukum
3. Fakultas Teknik dan Ilmu Komputer
4. Fakultas Ekonomi dan Bisnis Islam
5. Fakultas Dakwah
6. Fakultas Kesehatan
7. Pascasarjana

### KEUNGGULAN
- Integrasi ilmu pengetahuan dan nilai-nilai Islam
- Lingkungan kampus kondusif di pegunungan Wonosobo
- Biaya terjangkau dengan berbagai beasiswa
- Fasilitas lengkap: perpustakaan, laboratorium, masjid
""",

    "out_of_topic": """
## INFORMASI PMB UNSIQ

Saya adalah asisten virtual PMB UNSIQ yang membantu informasi seputar:
- Pendaftaran mahasiswa baru
- Program studi
- Biaya kuliah dan beasiswa
- Fasilitas kampus
- Persyaratan pendaftaran

Untuk pertanyaan di luar topik PMB, saya tidak dapat memberikan informasi.

### KONTAK PMB:
📞 (0286) 321873 | 📱 0857 7504 7504
"""
}

# =============================================================================
# SCENARIOS PER CATEGORY
# =============================================================================

SCENARIOS = {
    "alur_pendaftaran": [
        ("siswa_baru", "Tanya langkah-langkah pendaftaran online"),
        ("orang_tua", "Tanya prosedur pendaftaran untuk anak"),
        ("siswa_kerja", "Tanya alur daftar kelas ekstension"),
        ("calon_s2", "Tanya prosedur pendaftaran pascasarjana"),
        ("siswa_baru", "Tanya cara upload dokumen"),
        ("orang_tua", "Tanya cara bayar biaya pendaftaran"),
        ("siswa_baru", "Tanya cara cek hasil seleksi"),
        ("calon_s2", "Tanya tahapan daftar ulang"),
        ("siswa_kerja", "Lupa nomor pendaftaran, tanya cara recovery"),
        ("orang_tua", "Tanya dokumen apa saja yang harus disiapkan"),
    ],
    
    "informasi_umum": [
        ("siswa_baru", "Tanya jadwal pendaftaran gelombang 1"),
        ("orang_tua", "Tanya jam layanan PMB"),
        ("siswa_kerja", "Tanya kontak WhatsApp PMB"),
        ("calon_s2", "Tanya lokasi kampus UNSIQ"),
        ("siswa_baru", "Tanya deadline pendaftaran gelombang 2"),
        ("orang_tua", "Tanya biaya pendaftaran per gelombang"),
        ("siswa_baru", "Tanya akreditasi UNSIQ"),
        ("siswa_kerja", "Tanya email resmi PMB"),
        ("calon_s2", "Tanya jadwal gelombang 3"),
        ("siswa_baru", "Tanya alamat website PMB"),
    ],
    
    "profil_unsiq": [
        ("siswa_baru", "Tanya sejarah singkat UNSIQ"),
        ("orang_tua", "Tanya visi misi UNSIQ"),
        ("siswa_kerja", "Tanya fakultas apa saja di UNSIQ"),
        ("calon_s2", "Tanya keunggulan UNSIQ"),
        ("siswa_baru", "Tanya akreditasi institusi UNSIQ"),
        ("orang_tua", "Tanya lokasi kampus UNSIQ"),
        ("siswa_kerja", "Tanya tahun berdiri UNSIQ"),
        ("calon_s2", "Tanya program pascasarjana UNSIQ"),
        ("siswa_baru", "Tanya fasilitas kampus UNSIQ"),
        ("orang_tua", "Tanya suasana kampus UNSIQ"),
        ("siswa_baru", "Tanya keunikan UNSIQ dibanding kampus lain"),
        ("siswa_kerja", "Tanya kerjasama UNSIQ dengan industri"),
        ("calon_s2", "Tanya research center UNSIQ"),
        ("orang_tua", "Tanya keamanan kampus"),
        ("siswa_baru", "Tanya apakah UNSIQ pondok pesantren"),
    ],
    
    "out_of_topic": [
        ("siswa_baru", "Tanya harga laptop gaming terbaik"),
        ("orang_tua", "Tanya resep masakan padang"),
        ("siswa_kerja", "Tanya cara investasi saham"),
        ("siswa_baru", "Tanya kapan Indonesia merdeka"),
        ("orang_tua", "Tanya tempat wisata di Bali"),
    ]
}

SYSTEM_PROMPT = """Kamu adalah asisten PMB UNSIQ yang ramah dan helpful.
TUGAS: Generate percakapan multi-turn (2-4 giliran) antara user {persona} dengan asisten PMB.

PERSONA USER: {persona_desc}
SKENARIO: {scenario}

KONTEKS REFERENSI:
{context}

OUTPUT FORMAT (JSON array):
[
  {{"role": "user", "content": "pertanyaan user"}},
  {{"role": "model", "content": "jawaban asisten"}},
  ...
]

ATURAN:
1. Jadikan percakapan NATURAL dan RELEVAN dengan skenario
2. Jawaban harus AKURAT berdasarkan konteks
3. Gunakan bahasa sopan dan ramah
4. 2-4 giliran (user-model pairs)
5. HANYA output JSON array, tidak ada teks lain
"""

PERSONA_DESC = {
    "siswa_baru": "Siswa SMA/SMK yang baru lulus, excited tapi bingung prosedur",
    "orang_tua": "Orang tua yang ingin mendaftarkan anak, perlu penjelasan detail",
    "siswa_kerja": "Karyawan yang ingin kuliah sambil bekerja, waktu terbatas",
    "calon_s2": "Lulusan S1 yang ingin lanjut S2, fokus pada prosedur pascasarjana"
}

def generate_test_data(engine, category: str, num_samples: int) -> List[Dict]:
    """Generate test data for a specific category."""
    results = []
    scenarios = SCENARIOS[category]
    context = CONTEXTS[category]
    
    # Cycle through scenarios
    for i in tqdm(range(num_samples), desc=f"Generating {category}"):
        persona, scenario = scenarios[i % len(scenarios)]
        
        prompt = SYSTEM_PROMPT.format(
            persona=persona,
            persona_desc=PERSONA_DESC[persona],
            scenario=scenario,
            context=context
        )
        
        try:
            response = engine.generate_single(prompt, max_tokens=1024, temperature=0.7)
            
            # Parse JSON
            text = response.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            
            conversation = json.loads(text)
            
            if isinstance(conversation, list) and len(conversation) >= 2:
                # Build text field
                text_parts = []
                for turn in conversation:
                    role = turn.get("role", "")
                    content = turn.get("content", "")
                    if role == "user":
                        text_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
                    elif role == "model":
                        text_parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
                
                result = {
                    "id": f"test_{category}_{i:03d}",
                    "source": "synthetic_test",
                    "category": category,
                    "persona": persona,
                    "complexity": "medium",
                    "conversation": conversation,
                    "text": "\n".join(text_parts),
                    "num_turns": len(conversation)
                }
                results.append(result)
        except Exception as e:
            print(f"Error generating {category} sample {i}: {e}")
            continue
    
    return results


def main():
    if not HAS_VLLM:
        print("ERROR: vLLM is required")
        return
    
    print("="*60)
    print("GENERATE ADDITIONAL TEST DATA")
    print("="*60)
    
    # Initialize engine
    engine = VLLMEngine(model_name="google/gemma-3-4b-it")
    
    # Categories and how many NEW samples needed
    categories_to_generate = {
        "alur_pendaftaran": 7,   # 8 → 15
        "informasi_umum": 8,     # 7 → 15
        "profil_unsiq": 14,      # 1 → 15
        "out_of_topic": 3,       # 12 → 15
    }
    
    all_new_test_data = []
    
    for category, num_needed in categories_to_generate.items():
        print(f"\nGenerating {num_needed} samples for {category}...")
        new_data = generate_test_data(engine, category, num_needed)
        all_new_test_data.extend(new_data)
        print(f"  Generated {len(new_data)} samples")
    
    # Load existing test data
    with open("data/merged/multiturn_test.json", "r") as f:
        existing_test = json.load(f)
    
    print(f"\nExisting test samples: {len(existing_test)}")
    print(f"New test samples: {len(all_new_test_data)}")
    
    # Merge
    combined_test = existing_test + all_new_test_data
    
    # Save
    with open("data/merged/multiturn_test.json", "w") as f:
        json.dump(combined_test, f, ensure_ascii=False, indent=2)
    
    print(f"\nTotal test samples: {len(combined_test)}")
    
    # Show distribution
    from collections import Counter
    cats = Counter([i.get('category', 'unknown') for i in combined_test])
    print("\nFinal distribution:")
    for cat, count in sorted(cats.items()):
        status = "✅" if count >= 15 else "⚠️"
        print(f"  {cat}: {count} {status}")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
