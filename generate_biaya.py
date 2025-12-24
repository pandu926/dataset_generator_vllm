#!/usr/bin/env python3
"""
High-Quality Biaya (Fees/Cost) Multi-turn Dataset Generator
Generates 200 diverse conversations about university fees with FORMAL response style.

Usage: python generate_biaya.py
"""

import os
import json
import random
from typing import List, Dict
from tqdm import tqdm

SEED = 758
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
# CONTEXT: BIAYA PENDIDIKAN UNSIQ
# =============================================================================

CONTEXT_BIAYA = {
    "pendaftaran": """
## BIAYA PENDAFTARAN

### Gelombang 1 (November-Desember):
- D3 & S1: GRATIS (Rp 0)
- S2: GRATIS (Rp 0)

### Gelombang 2 (Januari-April):
- D3 & S1: Rp 250.000
- S2: Rp 300.000

### Gelombang 3 (Mei-Juli):
- D3 & S1: Rp 250.000
- S2: Rp 300.000

### Metode Pembayaran:
- Bank Jateng (Aplikasi/ATM)
- Finpay (Mobile payment)
- Gunakan NOMOR PENDAFTARAN sebagai referensi

### Keuntungan Gelombang 1:
- Biaya pendaftaran GRATIS
- Kuota paling banyak
- Bisa mulai kuliah lebih cepat
""",

    "semester": """
## BIAYA SEMESTER (ANGSURAN)

### Biaya Angsuran Pertama (Setelah Diterima):
- D3: Rp 745.000
- S1: Rp 745.000
- S2: Rp 1.100.000

### Sistem Pembayaran:
- Pembayaran via Bank Jateng/Finpay
- Gunakan Nomor Pendaftaran sebagai referensi
- Status update dalam 1-2 jam

### Cicilan Semester:
- Pembayaran bisa dicicil per semester
- Jadwal cicilan diinformasikan setelah reservasi NIM
- Keterlambatan dikenakan denda administrasi
""",

    "ukt": """
## UKT (UANG KULIAH TUNGGAL)

### Sistem UKT UNSIQ:
- UKT berdasarkan kemampuan ekonomi keluarga
- Ditetapkan setelah verifikasi data ekonomi
- Berlaku per semester

### Kategori UKT:
- UKT 1: Untuk keluarga kurang mampu (termurah)
- UKT 2: Untuk keluarga menengah bawah
- UKT 3: Untuk keluarga menengah
- UKT 4: Untuk keluarga menengah atas
- UKT 5: Untuk keluarga mampu (tertinggi)

### Cara Pengajuan UKT Rendah:
1. Isi formulir pengajuan UKT
2. Lampirkan bukti penghasilan orang tua
3. Lampirkan foto rumah dan kondisi ekonomi
4. Tunggu verifikasi (7-14 hari kerja)
""",

    "beasiswa": """
## BEASISWA UNSIQ

### Jenis Beasiswa:
1. Beasiswa Prestasi Akademik
2. Beasiswa Tahfidz Al-Qur'an
3. Beasiswa Kurang Mampu (BBM)
4. Beasiswa KIP-Kuliah
5. Beasiswa Santri Berprestasi

### Syarat Umum Beasiswa:
- Terdaftar sebagai mahasiswa aktif
- IPK minimal 3.0 (untuk beasiswa akademik)
- Tidak sedang menerima beasiswa lain
- Mengisi formulir pendaftaran beasiswa

### Beasiswa Tahfidz:
- Hafal minimal 5 juz: potongan 25%
- Hafal minimal 10 juz: potongan 50%
- Hafal 30 juz: GRATIS biaya kuliah

### KIP-Kuliah:
- Untuk keluarga kurang mampu
- Gratis biaya kuliah + uang saku bulanan
- Daftar melalui portal KIP-Kuliah Kemendikbud
""",

    "rincian_prodi": """
## RINCIAN BIAYA PER PRODI

### Fakultas Teknik dan Ilmu Komputer (FASTIKOM):
- Informatika: UKT Rp 3.500.000 - Rp 7.000.000/semester
- Sistem Informasi: UKT Rp 3.500.000 - Rp 7.000.000/semester

### Fakultas Ilmu Kesehatan (FIKES):
- Keperawatan: UKT Rp 4.000.000 - Rp 8.000.000/semester
- Farmasi: UKT Rp 4.500.000 - Rp 9.000.000/semester
- Kebidanan (D3): UKT Rp 3.500.000 - Rp 7.000.000/semester

### Fakultas Ekonomi dan Bisnis:
- Manajemen: UKT Rp 3.000.000 - Rp 6.000.000/semester
- Akuntansi: UKT Rp 3.000.000 - Rp 6.000.000/semester

### Fakultas Ilmu Tarbiyah dan Keguruan (FITK):
- PAI: UKT Rp 2.500.000 - Rp 5.000.000/semester
- PGMI: UKT Rp 2.500.000 - Rp 5.000.000/semester
- PBA: UKT Rp 2.500.000 - Rp 5.000.000/semester

### Pascasarjana (S2):
- Manajemen Pendidikan Islam: Rp 3.500.000 - Rp 7.000.000/semester
- Hukum Ekonomi Syariah: Rp 3.500.000 - Rp 7.000.000/semester
""",

    "cicilan": """
## SISTEM CICILAN

### Skema Cicilan:
- Pembayaran bisa dicicil maksimal 4x per semester
- Cicilan 1: Sebelum perkuliahan dimulai (minimal 50%)
- Cicilan 2: Bulan ke-2 perkuliahan
- Cicilan 3: Bulan ke-3 perkuliahan
- Cicilan 4: Sebelum UTS

### Syarat Pengajuan Cicilan:
1. Mengisi formulir pengajuan cicilan
2. Surat pernyataan kesanggupan bayar
3. Disetujui oleh bagian keuangan

### Denda Keterlambatan:
- Keterlambatan 1-7 hari: Denda Rp 50.000
- Keterlambatan 8-14 hari: Denda Rp 100.000
- Keterlambatan > 14 hari: Tidak bisa ikut UAS

### Cara Pengajuan:
- Datang ke bagian keuangan
- Atau hubungi via email: keuangan@unsiq.ac.id
""",

    "biaya_lain": """
## BIAYA LAIN-LAIN

### Biaya Tambahan:
- Praktikum Laboratorium: Rp 200.000 - Rp 500.000/semester (tergantung prodi)
- KKN: Rp 500.000 (sekali bayar)
- Wisuda: Rp 750.000 (termasuk toga dan ijazah)
- Legalisir Ijazah: Rp 10.000/lembar

### Biaya Khusus Fakultas Kesehatan:
- Praktek Klinik/RS: Rp 1.000.000 - Rp 2.000.000/semester
- Seragam Praktek: Rp 500.000 (sekali bayar)

### Fasilitas yang Sudah Termasuk UKT:
- Akses perpustakaan
- Akses wifi kampus
- Kartu mahasiswa (KTM)
- Akses sistem akademik online
- Fasilitas olahraga
""",

    "refund": """
## PENGEMBALIAN BIAYA (REFUND)

### Kebijakan Refund:
- Pengunduran diri sebelum kuliah dimulai: Refund 75%
- Pengunduran diri minggu 1-2: Refund 50%
- Pengunduran diri minggu 3-4: Refund 25%
- Pengunduran diri setelah 1 bulan: Tidak ada refund

### Syarat Pengajuan Refund:
1. Surat pernyataan mengundurkan diri
2. Kartu mahasiswa (jika sudah terbit)
3. Bukti pembayaran asli
4. Fotokopi KTP

### Proses Refund:
- Pengajuan ke bagian keuangan
- Verifikasi 7-14 hari kerja
- Transfer ke rekening dalam 30 hari kerja
""",

    "kontak": """
## KONTAK KEUANGAN

### Bagian Keuangan UNSIQ:
- Telepon: (0286) 321873 ext. 123
- WhatsApp: 0812-XXXX-XXXX
- Email: keuangan@unsiq.ac.id
- Jam Layanan: Senin-Jumat, 08:00-15:00 WIB

### Bagian PMB (untuk biaya pendaftaran):
- Telepon: (0286) 321873
- WhatsApp: 0857 7504 7504
- Email: humas@unsiq.ac.id

### Alamat:
Universitas Sains Al-Qur'an (UNSIQ)
Jl. KH. Hasyim Asy'ari Km. 03, Kalibebar
Kec. Mojotengah, Kab. Wonosobo
Jawa Tengah 56351
"""
}

# =============================================================================
# 200 SCENARIOS
# =============================================================================

SCENARIOS = [
    # === BIAYA PENDAFTARAN (25 scenarios) ===
    {"id": "BP01", "category": "pendaftaran", "scenario": "User tanya biaya pendaftaran Gelombang 1", "complexity": "direct"},
    {"id": "BP02", "category": "pendaftaran", "scenario": "User tanya biaya pendaftaran Gelombang 2", "complexity": "direct"},
    {"id": "BP03", "category": "pendaftaran", "scenario": "User tanya biaya pendaftaran Gelombang 3", "complexity": "direct"},
    {"id": "BP04", "category": "pendaftaran", "scenario": "User tanya perbedaan biaya pendaftaran D3/S1 vs S2", "complexity": "direct"},
    {"id": "BP05", "category": "pendaftaran", "scenario": "User tanya metode pembayaran pendaftaran", "complexity": "direct"},
    {"id": "BP06", "category": "pendaftaran", "scenario": "User tanya cara bayar via Bank Jateng", "complexity": "reasoning"},
    {"id": "BP07", "category": "pendaftaran", "scenario": "User tanya cara bayar via Finpay", "complexity": "reasoning"},
    {"id": "BP08", "category": "pendaftaran", "scenario": "User tanya referensi pembayaran yang benar", "complexity": "reasoning"},
    {"id": "BP09", "category": "pendaftaran", "scenario": "User sudah bayar tapi status belum berubah", "complexity": "edge_case"},
    {"id": "BP10", "category": "pendaftaran", "scenario": "User transfer pakai nomor rekening bukan nomor pendaftaran", "complexity": "edge_case"},
    {"id": "BP11", "category": "pendaftaran", "scenario": "User tanya keuntungan daftar Gelombang 1", "complexity": "reasoning"},
    {"id": "BP12", "category": "pendaftaran", "scenario": "User bingung pilih Gelombang mana yang paling hemat", "complexity": "reasoning"},
    {"id": "BP13", "category": "pendaftaran", "scenario": "User tanya apakah biaya pendaftaran bisa dicicil", "complexity": "direct"},
    {"id": "BP14", "category": "pendaftaran", "scenario": "User tanya deadline pembayaran pendaftaran", "complexity": "direct"},
    {"id": "BP15", "category": "pendaftaran", "scenario": "User tanya apakah biaya pendaftaran bisa refund", "complexity": "edge_case"},
    {"id": "BP16", "category": "pendaftaran", "scenario": "User salah transfer nominal pembayaran", "complexity": "edge_case"},
    {"id": "BP17", "category": "pendaftaran", "scenario": "User tanya apakah ada diskon pendaftaran", "complexity": "direct"},
    {"id": "BP18", "category": "pendaftaran", "scenario": "User dari luar kota tanya cara bayar online", "complexity": "reasoning"},
    {"id": "BP19", "category": "pendaftaran", "scenario": "User tanya apakah harus bayar langsung ke kampus", "complexity": "direct"},
    {"id": "BP20", "category": "pendaftaran", "scenario": "User tanya berapa lama konfirmasi pembayaran", "complexity": "direct"},
    {"id": "BP21", "category": "pendaftaran", "scenario": "User tanya cara cek status pembayaran pendaftaran", "complexity": "direct"},
    {"id": "BP22", "category": "pendaftaran", "scenario": "User mau bayar tapi tidak punya Bank Jateng", "complexity": "reasoning"},
    {"id": "BP23", "category": "pendaftaran", "scenario": "User tanya bisa bayar pakai transfer bank lain", "complexity": "direct"},
    {"id": "BP24", "category": "pendaftaran", "scenario": "User tanya biaya pendaftaran untuk mahasiswa pindahan", "complexity": "edge_case"},
    {"id": "BP25", "category": "pendaftaran", "scenario": "User tanya total biaya dari daftar sampai jadi mahasiswa", "complexity": "reasoning"},

    # === BIAYA SEMESTER (25 scenarios) ===
    {"id": "BS01", "category": "semester", "scenario": "User tanya biaya angsuran pertama D3", "complexity": "direct"},
    {"id": "BS02", "category": "semester", "scenario": "User tanya biaya angsuran pertama S1", "complexity": "direct"},
    {"id": "BS03", "category": "semester", "scenario": "User tanya biaya angsuran pertama S2", "complexity": "direct"},
    {"id": "BS04", "category": "semester", "scenario": "User tanya cara bayar biaya semester", "complexity": "reasoning"},
    {"id": "BS05", "category": "semester", "scenario": "User tanya deadline bayar angsuran pertama", "complexity": "direct"},
    {"id": "BS06", "category": "semester", "scenario": "User sudah bayar semester tapi status tidak update", "complexity": "edge_case"},
    {"id": "BS07", "category": "semester", "scenario": "User tanya apakah biaya semester sama setiap semester", "complexity": "direct"},
    {"id": "BS08", "category": "semester", "scenario": "User tanya kapan harus bayar semester berikutnya", "complexity": "direct"},
    {"id": "BS09", "category": "semester", "scenario": "User telat bayar semester tanya konsekuensinya", "complexity": "edge_case"},
    {"id": "BS10", "category": "semester", "scenario": "User tanya perbedaan biaya semester D3 dan S1", "complexity": "direct"},
    {"id": "BS11", "category": "semester", "scenario": "User mau cuti kuliah tanya biayanya", "complexity": "edge_case"},
    {"id": "BS12", "category": "semester", "scenario": "User tanya biaya semester untuk mahasiswa aktif organisasi", "complexity": "direct"},
    {"id": "BS13", "category": "semester", "scenario": "User tanya apa saja yang termasuk biaya semester", "complexity": "reasoning"},
    {"id": "BS14", "category": "semester", "scenario": "User tanya biaya semester naik setiap tahun atau tidak", "complexity": "direct"},
    {"id": "BS15", "category": "semester", "scenario": "User tidak mampu bayar semester penuh", "complexity": "edge_case"},
    {"id": "BS16", "category": "semester", "scenario": "User tanya cara konfirmasi pembayaran semester", "complexity": "direct"},
    {"id": "BS17", "category": "semester", "scenario": "User tanya bisa bayar semester lebih awal", "complexity": "direct"},
    {"id": "BS18", "category": "semester", "scenario": "User tanya waktu pembayaran semester paling lambat", "complexity": "direct"},
    {"id": "BS19", "category": "semester", "scenario": "User tanya apakah ada potongan bayar lunas di awal", "complexity": "reasoning"},
    {"id": "BS20", "category": "semester", "scenario": "User orang tuanya meninggal tanya keringanan biaya", "complexity": "edge_case"},
    {"id": "BS21", "category": "semester", "scenario": "User tanya biaya semester untuk kelas Ekstension", "complexity": "direct"},
    {"id": "BS22", "category": "semester", "scenario": "User tanya perbedaan biaya kelas Reguler dan Ekstension", "complexity": "reasoning"},
    {"id": "BS23", "category": "semester", "scenario": "User tanya biaya semester untuk semester pendek", "complexity": "edge_case"},
    {"id": "BS24", "category": "semester", "scenario": "User tanya apakah biaya semester bisa ditransfer dari luar negeri", "complexity": "edge_case"},
    {"id": "BS25", "category": "semester", "scenario": "User tanya total estimasi biaya kuliah sampai lulus", "complexity": "reasoning"},

    # === UKT (25 scenarios) ===
    {"id": "UK01", "category": "ukt", "scenario": "User tanya apa itu UKT", "complexity": "direct"},
    {"id": "UK02", "category": "ukt", "scenario": "User tanya cara penetapan UKT", "complexity": "reasoning"},
    {"id": "UK03", "category": "ukt", "scenario": "User tanya kategori UKT yang ada", "complexity": "direct"},
    {"id": "UK04", "category": "ukt", "scenario": "User tanya cara mengajukan UKT rendah", "complexity": "reasoning"},
    {"id": "UK05", "category": "ukt", "scenario": "User tanya dokumen untuk pengajuan UKT", "complexity": "direct"},
    {"id": "UK06", "category": "ukt", "scenario": "User keberatan dengan UKT yang ditetapkan", "complexity": "edge_case"},
    {"id": "UK07", "category": "ukt", "scenario": "User tanya cara mengajukan keringanan UKT", "complexity": "reasoning"},
    {"id": "UK08", "category": "ukt", "scenario": "User dari keluarga tidak mampu tanya opsi UKT", "complexity": "reasoning"},
    {"id": "UK09", "category": "ukt", "scenario": "User tanya lama proses verifikasi UKT", "complexity": "direct"},
    {"id": "UK10", "category": "ukt", "scenario": "User tanya kapan UKT diumumkan", "complexity": "direct"},
    {"id": "UK11", "category": "ukt", "scenario": "User tanya apakah UKT bisa berubah tiap semester", "complexity": "direct"},
    {"id": "UK12", "category": "ukt", "scenario": "User kondisi ekonomi berubah minta penyesuaian UKT", "complexity": "edge_case"},
    {"id": "UK13", "category": "ukt", "scenario": "User tanya perbedaan UKT 1 sampai 5", "complexity": "direct"},
    {"id": "UK14", "category": "ukt", "scenario": "User tanya range nominal UKT per kategori", "complexity": "direct"},
    {"id": "UK15", "category": "ukt", "scenario": "User tidak puas dengan hasil penetapan UKT", "complexity": "edge_case"},
    {"id": "UK16", "category": "ukt", "scenario": "User tanya cara banding UKT", "complexity": "reasoning"},
    {"id": "UK17", "category": "ukt", "scenario": "User tanya deadline pengajuan keringanan UKT", "complexity": "direct"},
    {"id": "UK18", "category": "ukt", "scenario": "User orang tua PHK minta keringanan UKT darurat", "complexity": "edge_case"},
    {"id": "UK19", "category": "ukt", "scenario": "User anak yatim piatu tanya kategori UKT", "complexity": "edge_case"},
    {"id": "UK20", "category": "ukt", "scenario": "User tanya bukti apa saja untuk UKT 1", "complexity": "reasoning"},
    {"id": "UK21", "category": "ukt", "scenario": "User tanya apakah foto rumah wajib untuk UKT", "complexity": "direct"},
    {"id": "UK22", "category": "ukt", "scenario": "User tanya format surat keterangan penghasilan", "complexity": "direct"},
    {"id": "UK23", "category": "ukt", "scenario": "User tanya siapa yang menentukan UKT", "complexity": "direct"},
    {"id": "UK24", "category": "ukt", "scenario": "User tanya apakah UKT berbeda tiap prodi", "complexity": "reasoning"},
    {"id": "UK25", "category": "ukt", "scenario": "User tanya perbedaan UKT UNSIQ dengan kampus lain", "complexity": "reasoning"},

    # === BEASISWA (30 scenarios) ===
    {"id": "BE01", "category": "beasiswa", "scenario": "User tanya jenis beasiswa yang tersedia", "complexity": "direct"},
    {"id": "BE02", "category": "beasiswa", "scenario": "User tanya syarat beasiswa prestasi akademik", "complexity": "direct"},
    {"id": "BE03", "category": "beasiswa", "scenario": "User tanya beasiswa Tahfidz Al-Qur'an", "complexity": "direct"},
    {"id": "BE04", "category": "beasiswa", "scenario": "User hafal 5 juz tanya potongan biaya", "complexity": "reasoning"},
    {"id": "BE05", "category": "beasiswa", "scenario": "User hafal 10 juz tanya potongan biaya", "complexity": "reasoning"},
    {"id": "BE06", "category": "beasiswa", "scenario": "User hafal 30 juz tanya keuntungannya", "complexity": "reasoning"},
    {"id": "BE07", "category": "beasiswa", "scenario": "User tanya beasiswa untuk keluarga kurang mampu", "complexity": "reasoning"},
    {"id": "BE08", "category": "beasiswa", "scenario": "User tanya cara daftar KIP-Kuliah", "complexity": "reasoning"},
    {"id": "BE09", "category": "beasiswa", "scenario": "User tanya syarat KIP-Kuliah", "complexity": "direct"},
    {"id": "BE10", "category": "beasiswa", "scenario": "User tanya benefit KIP-Kuliah", "complexity": "direct"},
    {"id": "BE11", "category": "beasiswa", "scenario": "User tanya beasiswa santri berprestasi", "complexity": "direct"},
    {"id": "BE12", "category": "beasiswa", "scenario": "User tanya kapan pendaftaran beasiswa dibuka", "complexity": "direct"},
    {"id": "BE13", "category": "beasiswa", "scenario": "User tanya IPK minimal untuk beasiswa", "complexity": "direct"},
    {"id": "BE14", "category": "beasiswa", "scenario": "User tanya apakah bisa dapat 2 beasiswa sekaligus", "complexity": "edge_case"},
    {"id": "BE15", "category": "beasiswa", "scenario": "User tanya cara mempertahankan beasiswa", "complexity": "reasoning"},
    {"id": "BE16", "category": "beasiswa", "scenario": "User IPK turun tanya konsekuensi beasiswa", "complexity": "edge_case"},
    {"id": "BE17", "category": "beasiswa", "scenario": "User tanya beasiswa untuk mahasiswa semester atas", "complexity": "reasoning"},
    {"id": "BE18", "category": "beasiswa", "scenario": "User maba tanya beasiswa yang bisa diambil", "complexity": "reasoning"},
    {"id": "BE19", "category": "beasiswa", "scenario": "User tanya dokumen pendaftaran beasiswa", "complexity": "direct"},
    {"id": "BE20", "category": "beasiswa", "scenario": "User tanya alur seleksi beasiswa", "complexity": "reasoning"},
    {"id": "BE21", "category": "beasiswa", "scenario": "User tanya pengumuman hasil seleksi beasiswa", "complexity": "direct"},
    {"id": "BE22", "category": "beasiswa", "scenario": "User tidak lolos beasiswa tanya opsi lain", "complexity": "edge_case"},
    {"id": "BE23", "category": "beasiswa", "scenario": "User tanya beasiswa dari pemerintah daerah", "complexity": "reasoning"},
    {"id": "BE24", "category": "beasiswa", "scenario": "User tanya beasiswa dari perusahaan/swasta", "complexity": "reasoning"},
    {"id": "BE25", "category": "beasiswa", "scenario": "User tanya perbedaan beasiswa penuh dan parsial", "complexity": "direct"},
    {"id": "BE26", "category": "beasiswa", "scenario": "User tanya kewajiban penerima beasiswa", "complexity": "direct"},
    {"id": "BE27", "category": "beasiswa", "scenario": "User tanya sanksi jika melanggar aturan beasiswa", "complexity": "edge_case"},
    {"id": "BE28", "category": "beasiswa", "scenario": "User ingin mundur dari beasiswa tanya prosedur", "complexity": "edge_case"},
    {"id": "BE29", "category": "beasiswa", "scenario": "User tanya beasiswa untuk program S2", "complexity": "direct"},
    {"id": "BE30", "category": "beasiswa", "scenario": "User tanya tips agar lolos seleksi beasiswa", "complexity": "reasoning"},

    # === RINCIAN BIAYA PRODI (30 scenarios) ===
    {"id": "RP01", "category": "rincian_prodi", "scenario": "User tanya UKT prodi Informatika", "complexity": "direct"},
    {"id": "RP02", "category": "rincian_prodi", "scenario": "User tanya UKT prodi Sistem Informasi", "complexity": "direct"},
    {"id": "RP03", "category": "rincian_prodi", "scenario": "User tanya UKT prodi Keperawatan", "complexity": "direct"},
    {"id": "RP04", "category": "rincian_prodi", "scenario": "User tanya UKT prodi Farmasi", "complexity": "direct"},
    {"id": "RP05", "category": "rincian_prodi", "scenario": "User tanya UKT prodi Kebidanan D3", "complexity": "direct"},
    {"id": "RP06", "category": "rincian_prodi", "scenario": "User tanya UKT prodi Manajemen", "complexity": "direct"},
    {"id": "RP07", "category": "rincian_prodi", "scenario": "User tanya UKT prodi Akuntansi", "complexity": "direct"},
    {"id": "RP08", "category": "rincian_prodi", "scenario": "User tanya UKT prodi PAI", "complexity": "direct"},
    {"id": "RP09", "category": "rincian_prodi", "scenario": "User tanya UKT prodi PGMI", "complexity": "direct"},
    {"id": "RP10", "category": "rincian_prodi", "scenario": "User tanya UKT prodi PBA", "complexity": "direct"},
    {"id": "RP11", "category": "rincian_prodi", "scenario": "User tanya UKT S2 Manajemen Pendidikan Islam", "complexity": "direct"},
    {"id": "RP12", "category": "rincian_prodi", "scenario": "User tanya UKT S2 Hukum Ekonomi Syariah", "complexity": "direct"},
    {"id": "RP13", "category": "rincian_prodi", "scenario": "User bandingkan UKT FASTIKOM dan FIKES", "complexity": "reasoning"},
    {"id": "RP14", "category": "rincian_prodi", "scenario": "User bandingkan UKT FEB dan FITK", "complexity": "reasoning"},
    {"id": "RP15", "category": "rincian_prodi", "scenario": "User tanya prodi dengan UKT termurah", "complexity": "reasoning"},
    {"id": "RP16", "category": "rincian_prodi", "scenario": "User tanya prodi dengan UKT termahal", "complexity": "reasoning"},
    {"id": "RP17", "category": "rincian_prodi", "scenario": "User tanya kenapa UKT FIKES lebih mahal", "complexity": "reasoning"},
    {"id": "RP18", "category": "rincian_prodi", "scenario": "User tanya range UKT terendah dan tertinggi", "complexity": "direct"},
    {"id": "RP19", "category": "rincian_prodi", "scenario": "User tanya total biaya kuliah prodi Farmasi", "complexity": "reasoning"},
    {"id": "RP20", "category": "rincian_prodi", "scenario": "User tanya total biaya kuliah prodi PAI", "complexity": "reasoning"},
    {"id": "RP21", "category": "rincian_prodi", "scenario": "User bingung pilih prodi berdasarkan biaya", "complexity": "reasoning"},
    {"id": "RP22", "category": "rincian_prodi", "scenario": "User tanya apakah UKT prodi kesehatan termasuk praktek", "complexity": "reasoning"},
    {"id": "RP23", "category": "rincian_prodi", "scenario": "User tanya biaya tambahan selain UKT per prodi", "complexity": "reasoning"},
    {"id": "RP24", "category": "rincian_prodi", "scenario": "User tanya UKT untuk kelas karyawan/Ekstension", "complexity": "direct"},
    {"id": "RP25", "category": "rincian_prodi", "scenario": "User tanya UKT prodi yang tidak ada di daftar", "complexity": "edge_case"},
    {"id": "RP26", "category": "rincian_prodi", "scenario": "User mau pindah prodi tanya perubahan UKT", "complexity": "edge_case"},
    {"id": "RP27", "category": "rincian_prodi", "scenario": "User tanya apakah UKT sama untuk semua mahasiswa seprodi", "complexity": "direct"},
    {"id": "RP28", "category": "rincian_prodi", "scenario": "User tanya faktor yang mempengaruhi UKT prodi", "complexity": "reasoning"},
    {"id": "RP29", "category": "rincian_prodi", "scenario": "User tanya proyeksi kenaikan UKT tahun depan", "complexity": "edge_case"},
    {"id": "RP30", "category": "rincian_prodi", "scenario": "User tanya prodi mana yang worth it dari segi biaya", "complexity": "reasoning"},

    # === CICILAN (25 scenarios) ===
    {"id": "CI01", "category": "cicilan", "scenario": "User tanya apakah bisa bayar cicilan", "complexity": "direct"},
    {"id": "CI02", "category": "cicilan", "scenario": "User tanya skema cicilan yang tersedia", "complexity": "direct"},
    {"id": "CI03", "category": "cicilan", "scenario": "User tanya maksimal berapa kali cicilan", "complexity": "direct"},
    {"id": "CI04", "category": "cicilan", "scenario": "User tanya jadwal pembayaran cicilan", "complexity": "direct"},
    {"id": "CI05", "category": "cicilan", "scenario": "User tanya persentase cicilan pertama", "complexity": "direct"},
    {"id": "CI06", "category": "cicilan", "scenario": "User tanya syarat pengajuan cicilan", "complexity": "direct"},
    {"id": "CI07", "category": "cicilan", "scenario": "User tanya cara mengajukan cicilan", "complexity": "reasoning"},
    {"id": "CI08", "category": "cicilan", "scenario": "User tanya dokumen untuk pengajuan cicilan", "complexity": "direct"},
    {"id": "CI09", "category": "cicilan", "scenario": "User telat bayar cicilan tanya denda", "complexity": "edge_case"},
    {"id": "CI10", "category": "cicilan", "scenario": "User tanya denda keterlambatan 1-7 hari", "complexity": "direct"},
    {"id": "CI11", "category": "cicilan", "scenario": "User tanya denda keterlambatan 8-14 hari", "complexity": "direct"},
    {"id": "CI12", "category": "cicilan", "scenario": "User tanya konsekuensi telat lebih dari 14 hari", "complexity": "edge_case"},
    {"id": "CI13", "category": "cicilan", "scenario": "User tidak bisa bayar cicilan tepat waktu", "complexity": "edge_case"},
    {"id": "CI14", "category": "cicilan", "scenario": "User tanya apakah cicilan dikenakan bunga", "complexity": "direct"},
    {"id": "CI15", "category": "cicilan", "scenario": "User tanya siapa yang menyetujui pengajuan cicilan", "complexity": "direct"},
    {"id": "CI16", "category": "cicilan", "scenario": "User tanya lama proses persetujuan cicilan", "complexity": "direct"},
    {"id": "CI17", "category": "cicilan", "scenario": "User pengajuan cicilan ditolak tanya alasan", "complexity": "edge_case"},
    {"id": "CI18", "category": "cicilan", "scenario": "User tanya apakah cicilan berlaku untuk semua biaya", "complexity": "reasoning"},
    {"id": "CI19", "category": "cicilan", "scenario": "User tanya kontak bagian keuangan untuk cicilan", "complexity": "direct"},
    {"id": "CI20", "category": "cicilan", "scenario": "User tanya apakah bisa perpanjang cicilan", "complexity": "edge_case"},
    {"id": "CI21", "category": "cicilan", "scenario": "User tanya apakah ada keringanan denda", "complexity": "edge_case"},
    {"id": "CI22", "category": "cicilan", "scenario": "User mau lunasi cicilan lebih cepat", "complexity": "direct"},
    {"id": "CI23", "category": "cicilan", "scenario": "User tanya reminder pembayaran cicilan", "complexity": "direct"},
    {"id": "CI24", "category": "cicilan", "scenario": "User lupa jadwal cicilan tanya cara cek", "complexity": "direct"},
    {"id": "CI25", "category": "cicilan", "scenario": "User tanya apa itu surat pernyataan kesanggupan bayar", "complexity": "direct"},

    # === BIAYA LAIN-LAIN (20 scenarios) ===
    {"id": "BL01", "category": "biaya_lain", "scenario": "User tanya biaya praktikum laboratorium", "complexity": "direct"},
    {"id": "BL02", "category": "biaya_lain", "scenario": "User tanya biaya KKN", "complexity": "direct"},
    {"id": "BL03", "category": "biaya_lain", "scenario": "User tanya biaya wisuda", "complexity": "direct"},
    {"id": "BL04", "category": "biaya_lain", "scenario": "User tanya biaya legalisir ijazah", "complexity": "direct"},
    {"id": "BL05", "category": "biaya_lain", "scenario": "User tanya biaya praktek klinik FIKES", "complexity": "direct"},
    {"id": "BL06", "category": "biaya_lain", "scenario": "User tanya biaya seragam praktek FIKES", "complexity": "direct"},
    {"id": "BL07", "category": "biaya_lain", "scenario": "User tanya fasilitas yang termasuk UKT", "complexity": "direct"},
    {"id": "BL08", "category": "biaya_lain", "scenario": "User tanya apakah wifi kampus gratis", "complexity": "direct"},
    {"id": "BL09", "category": "biaya_lain", "scenario": "User tanya apakah perpustakaan bayar", "complexity": "direct"},
    {"id": "BL10", "category": "biaya_lain", "scenario": "User tanya biaya pembuatan KTM", "complexity": "direct"},
    {"id": "BL11", "category": "biaya_lain", "scenario": "User tanya biaya fasilitas olahraga", "complexity": "direct"},
    {"id": "BL12", "category": "biaya_lain", "scenario": "User tanya biaya parkir kampus", "complexity": "direct"},
    {"id": "BL13", "category": "biaya_lain", "scenario": "User tanya biaya asrama atau kos", "complexity": "edge_case"},
    {"id": "BL14", "category": "biaya_lain", "scenario": "User tanya total estimasi biaya hidup per bulan", "complexity": "reasoning"},
    {"id": "BL15", "category": "biaya_lain", "scenario": "User tanya biaya buku dan perlengkapan kuliah", "complexity": "reasoning"},
    {"id": "BL16", "category": "biaya_lain", "scenario": "User tanya apakah ada biaya tersembunyi", "complexity": "reasoning"},
    {"id": "BL17", "category": "biaya_lain", "scenario": "User tanya biaya seminar atau workshop", "complexity": "direct"},
    {"id": "BL18", "category": "biaya_lain", "scenario": "User tanya biaya sertifikasi kompetensi", "complexity": "edge_case"},
    {"id": "BL19", "category": "biaya_lain", "scenario": "User tanya biaya skripsi atau tugas akhir", "complexity": "direct"},
    {"id": "BL20", "category": "biaya_lain", "scenario": "User tanya biaya perpanjangan studi", "complexity": "edge_case"},

    # === REFUND (10 scenarios) ===
    {"id": "RF01", "category": "refund", "scenario": "User tanya kebijakan refund biaya kuliah", "complexity": "direct"},
    {"id": "RF02", "category": "refund", "scenario": "User mundur sebelum kuliah tanya refund", "complexity": "reasoning"},
    {"id": "RF03", "category": "refund", "scenario": "User mundur minggu pertama tanya refund", "complexity": "reasoning"},
    {"id": "RF04", "category": "refund", "scenario": "User mundur setelah sebulan tanya refund", "complexity": "edge_case"},
    {"id": "RF05", "category": "refund", "scenario": "User tanya syarat pengajuan refund", "complexity": "direct"},
    {"id": "RF06", "category": "refund", "scenario": "User tanya dokumen untuk refund", "complexity": "direct"},
    {"id": "RF07", "category": "refund", "scenario": "User tanya proses dan lama refund", "complexity": "direct"},
    {"id": "RF08", "category": "refund", "scenario": "User tanya cara pengajuan refund", "complexity": "reasoning"},
    {"id": "RF09", "category": "refund", "scenario": "User tanya apakah biaya pendaftaran bisa direfund", "complexity": "direct"},
    {"id": "RF10", "category": "refund", "scenario": "User pindah kampus lain tanya refund", "complexity": "edge_case"},

    # === KONTAK KEUANGAN (10 scenarios) ===
    {"id": "KK01", "category": "kontak", "scenario": "User tanya kontak bagian keuangan", "complexity": "direct"},
    {"id": "KK02", "category": "kontak", "scenario": "User tanya jam layanan bagian keuangan", "complexity": "direct"},
    {"id": "KK03", "category": "kontak", "scenario": "User tanya email bagian keuangan", "complexity": "direct"},
    {"id": "KK04", "category": "kontak", "scenario": "User tanya alamat kantor keuangan", "complexity": "direct"},
    {"id": "KK05", "category": "kontak", "scenario": "User ingin konsultasi langsung soal biaya", "complexity": "reasoning"},
    {"id": "KK06", "category": "kontak", "scenario": "User komplain soal tagihan yang salah", "complexity": "edge_case"},
    {"id": "KK07", "category": "kontak", "scenario": "User tanya apakah bisa konsultasi via online", "complexity": "direct"},
    {"id": "KK08", "category": "kontak", "scenario": "User tanya perbedaan kontak PMB dan keuangan", "complexity": "reasoning"},
    {"id": "KK09", "category": "kontak", "scenario": "User di luar jam kerja butuh bantuan keuangan", "complexity": "edge_case"},
    {"id": "KK10", "category": "kontak", "scenario": "User tanya WhatsApp resmi bagian keuangan", "complexity": "direct"},
]

print(f"Total scenarios: {len(SCENARIOS)}")

# =============================================================================
# PROMPT TEMPLATES
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
❌ "Hai Kak! Wah keren banget mau daftar ya..."
✅ "Baik, berikut informasi biaya: 1. Biaya pendaftaran Rp 250.000 2. Angsuran pertama Rp 745.000"
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
    print("BIAYA (FEES) DATASET GENERATOR (FORMAL STYLE)")
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
    output_file = os.path.join(output_dir, "multiturn_biaya_200.json")
    
    # Generate
    generated_data = []
    batch_size = 10
    personas_list = list(PERSONAS.keys())
    total_scenarios = len(SCENARIOS)
    
    pbar = tqdm(total=total_scenarios, desc="Generating conversations", unit="conv")
    
    for batch_start in range(0, total_scenarios, batch_size):
        batch_scenarios = SCENARIOS[batch_start:batch_start+batch_size]
        
        # Build prompts with category-specific context
        prompts = []
        for scenario in batch_scenarios:
            persona_key = random.choice(personas_list)
            persona_desc = PERSONAS[persona_key]
            
            # Get category-specific context
            context = CONTEXT_BIAYA.get(scenario["category"], CONTEXT_BIAYA["pendaftaran"])
            
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
                    "instruction": f"Multi-turn conversation about UNSIQ fees - {scenario['category']}",
                    "input": "",
                    "output": json.dumps(conversation, ensure_ascii=False),
                    "text": "",
                    "category": "biaya",
                    "subcategory": scenario["category"],
                    "scenario": scenario["scenario"],
                    "complexity": scenario["complexity"],
                    "source": "synthetic_biaya_v1_formal"
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


if __name__ == "__main__":
    main()
