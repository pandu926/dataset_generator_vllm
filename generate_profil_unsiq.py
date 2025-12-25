#!/usr/bin/env python3
"""
UNSIQ Profile Multi-turn Dataset Generator
Generates diverse conversations about UNSIQ profile with FORMAL response style.
Uses SCENARIO-SPECIFIC context to stay within model token limits.

Usage: python generate_profil_unsiq.py
"""

import os
import json
import random
from typing import List, Dict
from tqdm import tqdm

SEED = 759
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
# SCENARIO-SPECIFIC CONTEXT MAPPING
# =============================================================================

CONTEXT_BY_STAGE = {
    "identitas": """
## IDENTITAS INSTITUSI UNSIQ

### Nama Resmi
Universitas Sains Al-Qur'an (UNSIQ) Jawa Tengah

### Lokasi
Wonosobo, Jawa Tengah, Indonesia

### Tahun Berdiri
2001 (berdasarkan SK Mendiknas No. 87/D/O/2001 tanggal 10 Juli 2001)

### Rektor
Dr. H. Zaenal Sukawi, M.A.

### Jenis Universitas
Transformatif, Humanis, dan Qur'ani

### Akreditasi
Terakreditasi sesuai standar BAN-PT

### Kontak
- Website: unsiq.ac.id
- Email: info@unsiq.ac.id
- Telepon: (0286) 321873
""",

    "sejarah": """
## SEJARAH UNSIQ

### Timeline Pembentukan:
| Tahun | Peristiwa |
|-------|-----------|
| 1987 | Pendirian Yayasan Ilmu-Ilmu Al-Qur'an (YIIQ) |
| 1988 | Berdiri Institut Ilmu Al-Qur'an (IIQ) - Fakultas Tarbiyah & Dakwah |
| 1996 | Berdiri Akademi Keperawatan (AKPER) - D3 Keperawatan & Kebidanan |
| 1999 | Berdiri STIE-YPIIQ - S1 Manajemen, Akuntansi, D3 Manajemen Informatika |
| 2001 | Peleburan 3 Lembaga menjadi UNSIQ |

### Asal-usul UNSIQ:
1. IIQ (Institut Ilmu Al-Qur'an) - 1988: Fakultas Tarbiyah & Dakwah
2. AKPER (Akademi Keperawatan) - 1996: D3 Keperawatan & Kebidanan
3. STIE (Sekolah Tinggi Ilmu Ekonomi) - 1999: S1 Manajemen, Akuntansi
""",

    "tokoh": """
## TOKOH PENDIRI UNSIQ

### K.H. Muntaha Al-Hafidz (Mbah Muntaha) - 1912-2004
Pendiri Utama dan Pengembang UNSIQ
- Militer: Pembuat bambu runcing melawan penjajah
- Politisi: Aktif dalam kepemimpinan masyarakat
- Pendidik: Berdedikasi tinggi
- Kyai Kharismatik: Tokoh agama berpengaruh luas

### Pendiri Bersama (4 Pilar):
| Pilar | Tokoh | Kontribusi |
|-------|-------|------------|
| Kyai & Pesantren | K.H. Muntaha | Nilai spiritual, ilmu agama |
| Birokrasi & Pemerintah | Drs. H. Poedjihardjo (Bupati) | Kebijakan, dukungan pemerintah |
| Dunia Usaha | A. Halimi | Finansial, infrastruktur |
| Akademisi | Drs. Karseno | Ilmu pengetahuan, metodologi |
""",

    "visi_misi": """
## VISI DAN MISI UNSIQ

### Visi:
Mengembangkan Sumber Daya Manusia (SDM) Unggul yang:
- Unggul dalam ilmu pengetahuan dan teknologi
- Unggul dalam budi pekerti dan nilai-nilai moral
- Unggul dalam nurani dan spiritualitas
- Sehat dan kuat secara fisik maupun mental
- Siap berkontribusi untuk Indonesia hebat dan peradaban dunia bermartabat

### Misi:
1. Menyelenggarakan pendidikan berbasis nilai Al-Qur'an terintegrasi ilmu pengetahuan modern
2. Mengembangkan mahasiswa menjadi insan berpengetahuan luas dan bermoral tinggi

### Karakteristik:
Universitas TRANSFORMATIF, HUMANIS, dan QUR'ANI
- Kuliah plus ngaji
- Jadi mahasiswa plus santri
- Qur'anisasi sains dengan saintifikasi Al-Qur'an
""",

    "fakultas": """
## FAKULTAS & PROGRAM STUDI UNSIQ

### 1. Fakultas Tarbiyah dan Ilmu Pendidikan (asal IIQ)
- Pendidikan Agama Islam (PAI)
- Komunikasi dan Penyiaran Islam (KPI)

### 2. Fakultas Ilmu Kesehatan (asal AKPER)
- D3 Keperawatan
- D3 Kebidanan
- S1 Keperawatan
- S1 Kebidanan

### 3. Fakultas Ekonomi dan Bisnis (asal STIE)
- S1 Manajemen
- S1 Akuntansi
- D3 Manajemen Informatika

### Jenjang Tersedia:
- D3 (Diploma Tiga) - 3 tahun
- S1 (Sarjana) - 4 tahun
- S2 (Pascasarjana) - 2-3 tahun
""",

    "mbkm": """
## KURIKULUM MERDEKA BELAJAR KAMPUS MERDEKA (MBKM)

### Model: Skema 5-1-2
- 5 = 5 semester di kampus dengan pembelajaran reguler
- 1 = 1 semester untuk aktivitas Merdeka Belajar
- 2 = 2 semester berikutnya atau disesuaikan

### 8 Program MBKM UNSIQ:
1. Magang/Praktik Kerja
2. Asistensi Mengajar di Satuan Pendidikan
3. Penelitian Akademik
4. Proyek Kemanusiaan
5. Wirausaha
6. Studi Independen
7. Pertukaran Pelajar (Mobilitas Mahasiswa)
8. Program Khusus Lainnya

### Implementasi:
- Mandiri oleh mahasiswa
- Fakultatif (atas permintaan)
- Universiter (program UNSIQ)
- Melalui hibah kompetisi dari Kemenag, Kemendikbud, dan instansi lain
""",

    "epistemologi": """
## EPISTEMOLOGI KEILMUAN UNSIQ

### Konsep: Syajarah Al-Qur'an
Harmonisasi, Sinergi, dan Integrasi

### Harmonis
Menyatukan berbagai perspektif ilmu dan tradisi dalam kesatuan seimbang.

### Sinergis
Menciptakan sinergi positif antara:
- Pesantren ↔ Universitas
- Tradisi ↔ Modernitas
- Spiritual ↔ Intelektual
- Al-Qur'an ↔ Sains

### Integratif
Mengintegrasikan secara menyeluruh:
- Kurikulum terintegrasi
- Pembelajaran holistik
- Luaran mahasiswa utuh secara intelectual dan spiritual
""",

    "keunggulan": """
## KEUNGGULAN UNSIQ

### 3 Keunggulan Lulusan:
1. Keunggulan Spesifik - Keahlian khusus di bidangnya
2. Keunggulan Kompetitif - Mampu bersaing di tingkat nasional/internasional
3. Keunggulan Komplementatif - Melengkapi kebutuhan masyarakat

### Identitas Unik:
- Unsiqers Learning Spirit (ULS)
- Harmoni pesantren + universitas modern
- Memadu tradisi dan modernitas

### Transformasi Digital:
- Sistem Data Terpadu
- Data-Driven Decision Making
- Continuous Improvement

### Kontribusi:
- Untuk Mahasiswa: Lulusan berkarakter kuat, adaptif
- Untuk Masyarakat: Pengabdian relevan, pemberdayaan
- Untuk Bangsa: Generasi pemimpin visioner
- Untuk Dunia: Perspektif qur'ani, kerjasama internasional
""",

    "kontak": """
## KONTAK & INFORMASI UNSIQ

### Website & Digital:
- Website Resmi: unsiq.ac.id
- Portal PMB: pmb.unsiq.ac.id
- Email: info@unsiq.ac.id

### Telepon:
- Kantor: (0286) 321873
- WhatsApp PMB: 0857 7504 7504

### Lokasi:
Wonosobo, Jawa Tengah, Indonesia

### Jam Layanan:
Senin-Jumat, 08:00-16:00 WIB
"""
}

# =============================================================================
# SCENARIOS (ALL ANSWERABLE FROM RAG)
# =============================================================================

SCENARIOS = [
    # ==========================================================================
    # IDENTITAS INSTITUSI (20 skenario)
    # ==========================================================================
    {"id": "PI01", "stage": "identitas", "scenario": "User tanya nama resmi dan tahun berdiri UNSIQ", "complexity": "direct"},
    {"id": "PI02", "stage": "identitas", "scenario": "User tanya lokasi kampus UNSIQ", "complexity": "direct"},
    {"id": "PI03", "stage": "identitas", "scenario": "User tanya siapa rektor UNSIQ saat ini", "complexity": "direct"},
    {"id": "PI04", "stage": "identitas", "scenario": "User tanya jenis universitas UNSIQ", "complexity": "direct"},
    {"id": "PI05", "stage": "identitas", "scenario": "User tanya status akreditasi UNSIQ", "complexity": "direct"},
    {"id": "PI06", "stage": "identitas", "scenario": "User tanya apakah UNSIQ universitas Islam", "complexity": "reasoning"},
    {"id": "PI07", "stage": "identitas", "scenario": "User tanya alamat lengkap kampus", "complexity": "direct"},
    {"id": "PI08", "stage": "identitas", "scenario": "User tanya apakah UNSIQ kampus swasta atau negeri", "complexity": "direct"},
    {"id": "PI09", "stage": "identitas", "scenario": "User tanya yayasan pengelola UNSIQ", "complexity": "reasoning"},
    {"id": "PI10", "stage": "identitas", "scenario": "User tanya luas area kampus UNSIQ", "complexity": "reasoning"},
    {"id": "PI11", "stage": "identitas", "scenario": "User tanya jumlah mahasiswa UNSIQ", "complexity": "reasoning"},
    {"id": "PI12", "stage": "identitas", "scenario": "User tanya jumlah dosen UNSIQ", "complexity": "reasoning"},
    {"id": "PI13", "stage": "identitas", "scenario": "User tanya apakah UNSIQ diakui Kemendikbud", "complexity": "reasoning"},
    {"id": "PI14", "stage": "identitas", "scenario": "User tanya kode PT UNSIQ", "complexity": "direct"},
    {"id": "PI15", "stage": "identitas", "scenario": "User tanya logo dan warna identitas UNSIQ", "complexity": "direct"},
    {"id": "PI16", "stage": "identitas", "scenario": "User tanya moto atau slogan UNSIQ", "complexity": "direct"},
    {"id": "PI17", "stage": "identitas", "scenario": "User tanya struktur organisasi UNSIQ", "complexity": "reasoning"},
    {"id": "PI18", "stage": "identitas", "scenario": "User tanya nama-nama Wakil Rektor", "complexity": "reasoning"},
    {"id": "PI19", "stage": "identitas", "scenario": "User tanya apakah UNSIQ terakreditasi internasional", "complexity": "reasoning"},
    {"id": "PI20", "stage": "identitas", "scenario": "User tanya ranking UNSIQ di Indonesia", "complexity": "reasoning"},

    # ==========================================================================
    # SEJARAH (20 skenario)
    # ==========================================================================
    {"id": "PS01", "stage": "sejarah", "scenario": "User tanya sejarah singkat berdirinya UNSIQ", "complexity": "reasoning"},
    {"id": "PS02", "stage": "sejarah", "scenario": "User tanya asal-usul 3 lembaga yang bergabung menjadi UNSIQ", "complexity": "reasoning"},
    {"id": "PS03", "stage": "sejarah", "scenario": "User tanya kapan IIQ, AKPER, dan STIE didirikan", "complexity": "direct"},
    {"id": "PS04", "stage": "sejarah", "scenario": "User tanya dasar hukum pendirian UNSIQ", "complexity": "direct"},
    {"id": "PS05", "stage": "sejarah", "scenario": "User tanya peran Bupati Wonosobo dalam pendirian UNSIQ", "complexity": "reasoning"},
    {"id": "PS06", "stage": "sejarah", "scenario": "User tanya tentang STIE-YPIIQ sebelum jadi UNSIQ", "complexity": "direct"},
    {"id": "PS07", "stage": "sejarah", "scenario": "User tanya sejarah Institut Ilmu Al-Qur'an (IIQ)", "complexity": "reasoning"},
    {"id": "PS08", "stage": "sejarah", "scenario": "User tanya sejarah Akademi Keperawatan (AKPER)", "complexity": "reasoning"},
    {"id": "PS09", "stage": "sejarah", "scenario": "User tanya kapan peleburan 3 lembaga menjadi UNSIQ", "complexity": "direct"},
    {"id": "PS10", "stage": "sejarah", "scenario": "User tanya latar belakang pendirian UNSIQ", "complexity": "reasoning"},
    {"id": "PS11", "stage": "sejarah", "scenario": "User tanya perkembangan UNSIQ dari 2001 sampai sekarang", "complexity": "reasoning"},
    {"id": "PS12", "stage": "sejarah", "scenario": "User tanya siapa rektor pertama UNSIQ", "complexity": "direct"},
    {"id": "PS13", "stage": "sejarah", "scenario": "User tanya berapa kali UNSIQ ganti rektor", "complexity": "reasoning"},
    {"id": "PS14", "stage": "sejarah", "scenario": "User tanya peran YPIIQ dalam sejarah UNSIQ", "complexity": "reasoning"},
    {"id": "PS15", "stage": "sejarah", "scenario": "User tanya milestone penting UNSIQ", "complexity": "reasoning"},
    {"id": "PS16", "stage": "sejarah", "scenario": "User tanya penghargaan yang pernah diraih UNSIQ", "complexity": "reasoning"},
    {"id": "PS17", "stage": "sejarah", "scenario": "User tanya hubungan UNSIQ dengan pesantren sekitar", "complexity": "reasoning"},
    {"id": "PS18", "stage": "sejarah", "scenario": "User tanya fakultas pertama yang ada di UNSIQ", "complexity": "direct"},
    {"id": "PS19", "stage": "sejarah", "scenario": "User tanya prodi baru yang dibuka setelah 2001", "complexity": "reasoning"},
    {"id": "PS20", "stage": "sejarah", "scenario": "User tanya rencana pengembangan UNSIQ ke depan", "complexity": "reasoning"},

    # ==========================================================================
    # TOKOH PENDIRI (15 skenario)
    # ==========================================================================
    {"id": "PT01", "stage": "tokoh", "scenario": "User tanya siapa pendiri utama UNSIQ", "complexity": "direct"},
    {"id": "PT02", "stage": "tokoh", "scenario": "User tanya tentang K.H. Muntaha Al-Hafidz", "complexity": "reasoning"},
    {"id": "PT03", "stage": "tokoh", "scenario": "User tanya konsep 4 pilar pendiri UNSIQ", "complexity": "reasoning"},
    {"id": "PT04", "stage": "tokoh", "scenario": "User tanya siapa saja tokoh pendiri selain Mbah Muntaha", "complexity": "direct"},
    {"id": "PT05", "stage": "tokoh", "scenario": "User tanya kontribusi Drs. H. Poedjihardjo", "complexity": "reasoning"},
    {"id": "PT06", "stage": "tokoh", "scenario": "User tanya siapa A. Halimi dalam sejarah UNSIQ", "complexity": "reasoning"},
    {"id": "PT07", "stage": "tokoh", "scenario": "User tanya peran Drs. Karseno sebagai akademisi", "complexity": "reasoning"},
    {"id": "PT08", "stage": "tokoh", "scenario": "User tanya tahun wafat Mbah Muntaha", "complexity": "direct"},
    {"id": "PT09", "stage": "tokoh", "scenario": "User tanya latar belakang militer Mbah Muntaha", "complexity": "reasoning"},
    {"id": "PT10", "stage": "tokoh", "scenario": "User tanya warisan pemikiran pendiri UNSIQ", "complexity": "reasoning"},
    {"id": "PT11", "stage": "tokoh", "scenario": "User tanya apakah ada museum atau memorial tokoh pendiri", "complexity": "reasoning"},
    {"id": "PT12", "stage": "tokoh", "scenario": "User tanya nilai-nilai yang diwariskan pendiri", "complexity": "reasoning"},
    {"id": "PT13", "stage": "tokoh", "scenario": "User tanya bagaimana 4 pilar bekerja sama", "complexity": "reasoning"},
    {"id": "PT14", "stage": "tokoh", "scenario": "User tanya inspirasi pendirian UNSIQ", "complexity": "reasoning"},
    {"id": "PT15", "stage": "tokoh", "scenario": "User tanya filosofi pendidikan Mbah Muntaha", "complexity": "reasoning"},

    # ==========================================================================
    # VISI MISI (20 skenario)
    # ==========================================================================
    {"id": "PV01", "stage": "visi_misi", "scenario": "User tanya visi UNSIQ", "complexity": "direct"},
    {"id": "PV02", "stage": "visi_misi", "scenario": "User tanya misi UNSIQ", "complexity": "direct"},
    {"id": "PV03", "stage": "visi_misi", "scenario": "User tanya karakteristik universitas UNSIQ", "complexity": "reasoning"},
    {"id": "PV04", "stage": "visi_misi", "scenario": "User tanya apa maksud transformatif, humanis, qur'ani", "complexity": "reasoning"},
    {"id": "PV05", "stage": "visi_misi", "scenario": "User tanya maksud slogan 'Kuliah Plus Ngaji'", "complexity": "reasoning"},
    {"id": "PV06", "stage": "visi_misi", "scenario": "User tanya tentang saintifikasi Al-Qur'an", "complexity": "reasoning"},
    {"id": "PV07", "stage": "visi_misi", "scenario": "User tanya makna qur'anisasi sains", "complexity": "reasoning"},
    {"id": "PV08", "stage": "visi_misi", "scenario": "User tanya tujuan pendidikan di UNSIQ", "complexity": "reasoning"},
    {"id": "PV09", "stage": "visi_misi", "scenario": "User tanya nilai-nilai yang dianut UNSIQ", "complexity": "reasoning"},
    {"id": "PV10", "stage": "visi_misi", "scenario": "User tanya target lulusan UNSIQ", "complexity": "reasoning"},
    {"id": "PV11", "stage": "visi_misi", "scenario": "User tanya apa itu mahasiswa plus santri", "complexity": "reasoning"},
    {"id": "PV12", "stage": "visi_misi", "scenario": "User tanya strategi pencapaian visi UNSIQ", "complexity": "reasoning"},
    {"id": "PV13", "stage": "visi_misi", "scenario": "User tanya komitmen UNSIQ untuk Indonesia", "complexity": "reasoning"},
    {"id": "PV14", "stage": "visi_misi", "scenario": "User tanya peran UNSIQ dalam pendidikan Islam", "complexity": "reasoning"},
    {"id": "PV15", "stage": "visi_misi", "scenario": "User tanya bagaimana UNSIQ menggabungkan tradisi dan modernitas", "complexity": "reasoning"},
    {"id": "PV16", "stage": "visi_misi", "scenario": "User tanya apakah visi UNSIQ sudah tercapai", "complexity": "reasoning"},
    {"id": "PV17", "stage": "visi_misi", "scenario": "User tanya rencana strategis UNSIQ 5 tahun ke depan", "complexity": "reasoning"},
    {"id": "PV18", "stage": "visi_misi", "scenario": "User tanya program unggulan untuk mencapai visi", "complexity": "reasoning"},
    {"id": "PV19", "stage": "visi_misi", "scenario": "User tanya indikator keberhasilan misi UNSIQ", "complexity": "reasoning"},
    {"id": "PV20", "stage": "visi_misi", "scenario": "User tanya perbedaan visi UNSIQ dengan kampus lain", "complexity": "reasoning"},

    # ==========================================================================
    # FAKULTAS & PRODI (25 skenario)
    # ==========================================================================
    {"id": "PF01", "stage": "fakultas", "scenario": "User tanya fakultas apa saja yang ada di UNSIQ", "complexity": "direct"},
    {"id": "PF02", "stage": "fakultas", "scenario": "User tanya program studi di Fakultas Kesehatan", "complexity": "direct"},
    {"id": "PF03", "stage": "fakultas", "scenario": "User tanya program studi di Fakultas Ekonomi", "complexity": "direct"},
    {"id": "PF04", "stage": "fakultas", "scenario": "User tanya jenjang pendidikan yang tersedia di UNSIQ", "complexity": "direct"},
    {"id": "PF05", "stage": "fakultas", "scenario": "User tertarik prodi Keperawatan, tanya informasinya", "complexity": "reasoning"},
    {"id": "PF06", "stage": "fakultas", "scenario": "User tanya apakah ada jurusan KPI", "complexity": "direct"},
    {"id": "PF07", "stage": "fakultas", "scenario": "User tanya jurusan manajemen masuk fakultas apa", "complexity": "direct"},
    {"id": "PF08", "stage": "fakultas", "scenario": "User tanya apakah ada program pascasarjana (S2)", "complexity": "direct"},
    {"id": "PF09", "stage": "fakultas", "scenario": "User tanya prodi PAI untuk apa", "complexity": "reasoning"},
    {"id": "PF10", "stage": "fakultas", "scenario": "User tanya prospek kerja lulusan FIKES", "complexity": "reasoning"},
    {"id": "PF11", "stage": "fakultas", "scenario": "User tanya prodi mana yang paling diminati", "complexity": "reasoning"},
    {"id": "PF12", "stage": "fakultas", "scenario": "User tanya akreditasi masing-masing prodi", "complexity": "reasoning"},
    {"id": "PF13", "stage": "fakultas", "scenario": "User tanya kurikulum prodi Manajemen", "complexity": "reasoning"},
    {"id": "PF14", "stage": "fakultas", "scenario": "User tanya fasilitas Fakultas Kesehatan", "complexity": "reasoning"},
    {"id": "PF15", "stage": "fakultas", "scenario": "User tanya dosen-dosen di Fakultas Tarbiyah", "complexity": "reasoning"},
    {"id": "PF16", "stage": "fakultas", "scenario": "User tanya kapasitas mahasiswa per prodi", "complexity": "reasoning"},
    {"id": "PF17", "stage": "fakultas", "scenario": "User tanya prodi D3 apa saja yang ada", "complexity": "direct"},
    {"id": "PF18", "stage": "fakultas", "scenario": "User tanya prodi S1 apa saja yang ada", "complexity": "direct"},
    {"id": "PF19", "stage": "fakultas", "scenario": "User tanya prodi S2 apa saja yang ada", "complexity": "direct"},
    {"id": "PF20", "stage": "fakultas", "scenario": "User tanya prodi baru yang akan dibuka", "complexity": "reasoning"},
    {"id": "PF21", "stage": "fakultas", "scenario": "User tanya bedanya prodi PAI dan KPI", "complexity": "reasoning"},
    {"id": "PF22", "stage": "fakultas", "scenario": "User tanya prodi yang cocok untuk calon guru", "complexity": "reasoning"},
    {"id": "PF23", "stage": "fakultas", "scenario": "User tanya prodi yang cocok untuk calon perawat", "complexity": "reasoning"},
    {"id": "PF24", "stage": "fakultas", "scenario": "User tanya prodi yang cocok untuk bisnis", "complexity": "reasoning"},
    {"id": "PF25", "stage": "fakultas", "scenario": "User tanya dekan masing-masing fakultas", "complexity": "reasoning"},

    # ==========================================================================
    # MBKM (15 skenario)
    # ==========================================================================
    {"id": "PM01", "stage": "mbkm", "scenario": "User tanya apa itu program MBKM di UNSIQ", "complexity": "direct"},
    {"id": "PM02", "stage": "mbkm", "scenario": "User tanya skema 5-1-2 dalam MBKM", "complexity": "reasoning"},
    {"id": "PM03", "stage": "mbkm", "scenario": "User tanya 8 program MBKM yang tersedia", "complexity": "direct"},
    {"id": "PM04", "stage": "mbkm", "scenario": "User tanya cara mengikuti program magang MBKM", "complexity": "reasoning"},
    {"id": "PM05", "stage": "mbkm", "scenario": "User tanya apakah MBKM wajib atau opsional", "complexity": "reasoning"},
    {"id": "PM06", "stage": "mbkm", "scenario": "User tanya contoh kegiatan proyek kemanusiaan di MBKM", "complexity": "reasoning"},
    {"id": "PM07", "stage": "mbkm", "scenario": "User tanya syarat mengikuti MBKM", "complexity": "reasoning"},
    {"id": "PM08", "stage": "mbkm", "scenario": "User tanya berapa SKS yang dikonversi dari MBKM", "complexity": "reasoning"},
    {"id": "PM09", "stage": "mbkm", "scenario": "User tanya partner MBKM UNSIQ", "complexity": "reasoning"},
    {"id": "PM10", "stage": "mbkm", "scenario": "User tanya program pertukaran pelajar MBKM", "complexity": "reasoning"},
    {"id": "PM11", "stage": "mbkm", "scenario": "User tanya studi independen di MBKM", "complexity": "reasoning"},
    {"id": "PM12", "stage": "mbkm", "scenario": "User tanya program wirausaha di MBKM", "complexity": "reasoning"},
    {"id": "PM13", "stage": "mbkm", "scenario": "User tanya hibah kompetisi untuk MBKM", "complexity": "reasoning"},
    {"id": "PM14", "stage": "mbkm", "scenario": "User tanya testimoni mahasiswa yang ikut MBKM", "complexity": "reasoning"},
    {"id": "PM15", "stage": "mbkm", "scenario": "User tanya keuntungan mengikuti MBKM", "complexity": "reasoning"},

    # ==========================================================================
    # EPISTEMOLOGI KEILMUAN (10 skenario)
    # ==========================================================================
    {"id": "PE01", "stage": "epistemologi", "scenario": "User tanya pendekatan keilmuan UNSIQ", "complexity": "reasoning"},
    {"id": "PE02", "stage": "epistemologi", "scenario": "User tanya apa itu Syajarah Al-Qur'an", "complexity": "reasoning"},
    {"id": "PE03", "stage": "epistemologi", "scenario": "User tanya konsep harmonis-sinergis-integratif", "complexity": "reasoning"},
    {"id": "PE04", "stage": "epistemologi", "scenario": "User tanya hubungan pesantren dan universitas di UNSIQ", "complexity": "reasoning"},
    {"id": "PE05", "stage": "epistemologi", "scenario": "User tanya integrasi kurikulum dengan nilai Islam", "complexity": "reasoning"},
    {"id": "PE06", "stage": "epistemologi", "scenario": "User tanya sinergi tradisi dan modernitas", "complexity": "reasoning"},
    {"id": "PE07", "stage": "epistemologi", "scenario": "User tanya pembelajaran holistik di UNSIQ", "complexity": "reasoning"},
    {"id": "PE08", "stage": "epistemologi", "scenario": "User tanya luaran mahasiswa yang utuh", "complexity": "reasoning"},
    {"id": "PE09", "stage": "epistemologi", "scenario": "User tanya riset berbasis Al-Qur'an", "complexity": "reasoning"},
    {"id": "PE10", "stage": "epistemologi", "scenario": "User tanya metodologi keilmuan UNSIQ", "complexity": "reasoning"},

    # ==========================================================================
    # KEUNGGULAN (15 skenario)
    # ==========================================================================
    {"id": "PK01", "stage": "keunggulan", "scenario": "User tanya keunggulan kuliah di UNSIQ", "complexity": "reasoning"},
    {"id": "PK02", "stage": "keunggulan", "scenario": "User tanya 3 keunggulan lulusan UNSIQ", "complexity": "direct"},
    {"id": "PK03", "stage": "keunggulan", "scenario": "User tanya apa yang membedakan UNSIQ dengan kampus lain", "complexity": "reasoning"},
    {"id": "PK04", "stage": "keunggulan", "scenario": "User tanya kontribusi UNSIQ untuk masyarakat", "complexity": "reasoning"},
    {"id": "PK05", "stage": "keunggulan", "scenario": "User tanya apa itu Unsiqers Learning Spirit (ULS)", "complexity": "direct"},
    {"id": "PK06", "stage": "keunggulan", "scenario": "User tanya apakah sistem data UNSIQ sudah terintegrasi", "complexity": "direct"},
    {"id": "PK07", "stage": "keunggulan", "scenario": "User tanya transformasi digital UNSIQ", "complexity": "reasoning"},
    {"id": "PK08", "stage": "keunggulan", "scenario": "User tanya keunggulan spesifik lulusan", "complexity": "reasoning"},
    {"id": "PK09", "stage": "keunggulan", "scenario": "User tanya keunggulan kompetitif lulusan", "complexity": "reasoning"},
    {"id": "PK10", "stage": "keunggulan", "scenario": "User tanya keunggulan komplementatif lulusan", "complexity": "reasoning"},
    {"id": "PK11", "stage": "keunggulan", "scenario": "User tanya kontribusi UNSIQ untuk bangsa", "complexity": "reasoning"},
    {"id": "PK12", "stage": "keunggulan", "scenario": "User tanya kerjasama internasional UNSIQ", "complexity": "reasoning"},
    {"id": "PK13", "stage": "keunggulan", "scenario": "User tanya fasilitas unggulan UNSIQ", "complexity": "reasoning"},
    {"id": "PK14", "stage": "keunggulan", "scenario": "User tanya prestasi mahasiswa UNSIQ", "complexity": "reasoning"},
    {"id": "PK15", "stage": "keunggulan", "scenario": "User tanya alumni sukses UNSIQ", "complexity": "reasoning"},

    # ==========================================================================
    # KONTAK & LAYANAN (10 skenario)
    # ==========================================================================
    {"id": "PC01", "stage": "kontak", "scenario": "User tanya website resmi UNSIQ", "complexity": "direct"},
    {"id": "PC02", "stage": "kontak", "scenario": "User tanya nomor telepon UNSIQ", "complexity": "direct"},
    {"id": "PC03", "stage": "kontak", "scenario": "User tanya email UNSIQ", "complexity": "direct"},
    {"id": "PC04", "stage": "kontak", "scenario": "User tanya jam operasional layanan UNSIQ", "complexity": "direct"},
    {"id": "PC05", "stage": "kontak", "scenario": "User tanya nomor WhatsApp PMB", "complexity": "direct"},
    {"id": "PC06", "stage": "kontak", "scenario": "User tanya portal PMB online", "complexity": "direct"},
    {"id": "PC07", "stage": "kontak", "scenario": "User tanya cara menghubungi fakultas tertentu", "complexity": "reasoning"},
    {"id": "PC08", "stage": "kontak", "scenario": "User tanya alamat email pengaduan", "complexity": "reasoning"},
    {"id": "PC09", "stage": "kontak", "scenario": "User tanya social media resmi UNSIQ", "complexity": "direct"},
    {"id": "PC10", "stage": "kontak", "scenario": "User tanya cara berkunjung ke kampus", "complexity": "reasoning"},
]

print(f"Total scenarios: {len(SCENARIOS)}")

# =============================================================================
# PROMPT TEMPLATES - FORMAL STYLE
# =============================================================================

SYSTEM_PROMPT = """You are an expert Synthetic Data Generator for UNSIQ (Universitas Sains Al-Qur'an).
Generate HIGH-QUALITY, REALISTIC MULTI-TURN conversations for training a Customer Service AI.

STRICT RULES:
1. **CONTEXT**: Use ONLY facts from the provided context. Do NOT hallucinate.
2. **USER STYLE**: Can be casual OR formal based on persona.
3. **AI RESPONSE STYLE** (CRITICAL):
   - Professional, formal, helpful
   - Natural formal Indonesian: "Baik,", "Tentu,", "Berikut informasinya."
   - CONCISE - singkat, padat, jelas, tidak bertele-tele
   - DO NOT USE: "Kak", "Nih", "Sip", "Oke deh", "Halo!", "Hai!"
   - Use "Anda" not "kamu" or "Kakak"
   - Use numbered lists for multiple items
4. **THOUGHT**: Include reasoning: "1. Analyze: ... 2. Retrieve: ... 3. Answer: ..."
5. **TURNS**: Generate 3-4 Q&A pairs.
6. **FORMAT**: Valid JSON list only.

EXAMPLE:
❌ "Hai Kak! Wah keren banget mau tanya ya..."
✅ "Baik, berikut informasinya: 1. UNSIQ berdiri tahun 2001 2. Lokasi di Wonosobo, Jawa Tengah"
"""

USER_PROMPT_TEMPLATE = """
CONTEXT:
{context}

SCENARIO: {scenario}
PERSONA: {persona_name} - {persona_desc}

Generate a 3-4 turn conversation. User asks based on scenario/persona. AI responds formally and concisely.

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
    print("PROFIL UNSIQ DATASET GENERATOR (FORMAL STYLE)")
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
    output_file = os.path.join(output_dir, "multiturn_profil_unsiq.json")
    
    # Generate
    generated_data = []
    batch_size = 10
    personas_list = list(PERSONAS.keys())
    total_scenarios = len(SCENARIOS)
    
    pbar = tqdm(total=total_scenarios, desc="Generating conversations", unit="conv")
    
    for batch_start in range(0, total_scenarios, batch_size):
        batch_scenarios = SCENARIOS[batch_start:batch_start+batch_size]
        
        # Build prompts with SCENARIO-SPECIFIC context
        prompts = []
        for scenario in batch_scenarios:
            persona_key = random.choice(personas_list)
            persona_desc = PERSONAS[persona_key]
            
            # Get stage-specific context (SHORT!)
            context = CONTEXT_BY_STAGE.get(scenario["stage"], CONTEXT_BY_STAGE["identitas"])
            
            prompt = USER_PROMPT_TEMPLATE.format(
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
                    "instruction": f"Multi-turn conversation about UNSIQ profile - {scenario['stage']}",
                    "input": "",
                    "output": json.dumps(conversation, ensure_ascii=False),
                    "text": "",
                    "category": "profil_unsiq",
                    "stage": scenario["stage"],
                    "scenario": scenario["scenario"],
                    "complexity": scenario["complexity"],
                    "source": "synthetic_profil_v1_formal"
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
    
    # Also create clean version
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
    
    clean_file = os.path.join(output_dir, "multiturn_profil_unsiq_clean.json")
    with open(clean_file, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)
    
    print(f"Clean version saved to: {clean_file}")


if __name__ == "__main__":
    main()
