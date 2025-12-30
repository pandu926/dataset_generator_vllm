#!/usr/bin/env python3
"""
SNK (Syarat dan Ketentuan) Multi-turn Dataset Generator
Generates 300 diverse conversations about registration requirements
"""

import os
import json
import random
from typing import List, Dict
from tqdm import tqdm

SEED = 758
random.seed(SEED)

from src.llm_multiturn_generator import MultiTurnGenerator, PERSONAS
try:
    from src.vllm_engine import VLLMEngine
    HAS_VLLM = True
except ImportError:
    print("Warning: vLLM not found.")
    HAS_VLLM = False

# =============================================================================
# CONTEXT: SYARAT & KETENTUAN PENDAFTARAN UNSIQ
# =============================================================================

CONTEXT_SNK = {
    "timeline": """
## TIMELINE & JADWAL PENDAFTARAN

### D3 & S1 - Gelombang 1
- Status: GRATIS (Biaya Pendaftaran)
- Periode: 1 November 2024 - 31 Desember 2024
- Deadline Ujian: 31 Desember 2024
- Deadline Bayar Semester 1: 8 Januari 2025

### D3 & S1 - Gelombang 2
- Status: Rp 250.000
- Periode: 1 Januari 2025 - 30 April 2025
- Deadline Ujian: 30 April 2025
- Deadline Bayar Semester 1: 8 Mei 2025

### D3 & S1 - Gelombang 3
- Status: Rp 250.000
- Periode: 1 Mei 2025 - 31 Agustus 2025
- Deadline Ujian: 31 Agustus 2025
- Deadline Bayar Semester 1: 7 September 2025

### S2 - Gelombang 1
- Status: GRATIS
- Periode: 1 November 2025 - 31 Desember 2025
- Bayar Semester 1: Rp 1.100.000 (minimal)

### S2 - Gelombang 2 & 3
- Status: Rp 300.000
- Periode sama dengan D3/S1
""",

    "dokumen": """
## DOKUMEN YANG DIPERLUKAN

### 1. Formulir Pendaftaran Online
- Isi di portal PMB: https://pmb.unsiq.ac.id/
- Simpan Nomor Pendaftaran dan PIN/Password

### 2. Pembayaran Biaya Pendaftaran
- Melalui Bank Jateng atau Finpay
- Gunakan NOMOR PENDAFTARAN sebagai referensi
- Gelombang 1: GRATIS
- Gelombang 2 & 3: Rp 250.000 (D3/S1), Rp 300.000 (S2)

### 3. Scan Ijazah / Surat Kelulusan
- Format: JPG
- Ukuran: 2-5 MB
- Lulusan 2025 bisa pakai surat keterangan aktif dulu

### 4. Scan Kartu Keluarga (KK)
- Format: JPG
- Harus masih aktif dan jelas

### 5. Foto Formal Berwarna
- Ukuran: 250x350 pixel
- Background putih
- Gaya formal

### 6. Surat Keterangan Sehat (Khusus FIKES)
- Dari RS/Puskesmas/Klinik
- Isi: sehat, tinggi, berat, tes buta warna (pria)
""",

    "proses": """
## PROSES PENDAFTARAN STEP-BY-STEP

### Langkah 1: Isi Formulir Online
- Buka pmb.unsiq.ac.id
- Pilih jenjang dan prodi
- Simpan nomor pendaftaran & PIN

### Langkah 2: Bayar Pendaftaran (Kecuali Gel 1)
- Transfer via Bank Jateng/Finpay
- Pakai nomor pendaftaran sebagai referensi
- Simpan bukti transfer

### Langkah 3: Login & Lengkapi Biodata
- Login dengan nomor pendaftaran + PIN
- Isi biodata diri, keluarga, pendidikan

### Langkah 4: Upload Dokumen
- Ijazah/SKL, KK, Foto formal
- Surat kesehatan (khusus FIKES)

### Langkah 5: Tunggu Verifikasi (2x24 Jam)
- Tim PMB verifikasi dokumen
- Pantau portal untuk status

### Langkah 6: Ujian Placement Test Online
- Bisa kapan saja sebelum deadline
- Hasil langsung keluar

### Langkah 7: Bayar Semester 1
- D3/S1: Minimal Rp 745.000
- S2: Minimal Rp 1.100.000

### Langkah 8: Reservasi NIM
- Setelah bayar semester 1
- Resmi jadi mahasiswa UNSIQ
""",

    "persyaratan_khusus": """
## PERSYARATAN KHUSUS

### Tinggi Badan - Fakultas Ilmu Kesehatan
- Laki-laki: Minimal 155 cm
- Perempuan: Minimal 150 cm
- Jika kurang, tidak bisa diterima

### S1 Keperawatan Kelas Karyawan
- Harus sudah lulus D3 Keperawatan
- Tidak bisa dari SMA langsung

### Verifikasi Dokumen
- Proses 2x24 jam
- Jika ada kekurangan akan diberitahu
""",

    "ketentuan": """
## KETENTUAN & KONSEKUENSI

### Ketentuan Pembayaran
- Uang pendaftaran TIDAK dapat diambil kembali
- Gelombang 1: Gratis
- Gelombang 2 & 3: Rp 250K (D3/S1) atau Rp 300K (S2)
- Semester 1: Minimal Rp 745K (D3/S1) atau Rp 1.1jt (S2)

### Ketentuan Tidak Hadir
- Tidak ikut ujian sesuai jadwal: Dianggap MENGUNDURKAN DIRI
- Diterima tapi tidak herregistrasi: Dianggap MENGUNDURKAN DIRI
- Tidak lapor ke Bagian Registrasi: Dianggap MENGUNDURKAN DIRI
- Tidak kuliah sampai akhir semester 1: Dianggap MENGUNDURKAN DIRI
""",

    "pindah_prodi": """
## PINDAH PROGRAM STUDI

### Belum Placement Test
- Waktu: Kapan saja sebelum ujian
- Cara: Ubah sendiri di menu Biodata
- Biaya: Gratis

### Sudah Placement Test (belum reservasi NIM)
- Cara: Hubungi PMB untuk reset ujian
- Biaya: Gratis
- Setelah reset, ubah prodi lalu ujian ulang

### Sudah Reservasi NIM
- Waktu: Maksimal 7 hari setelah kuliah perdana
- Cara: Surat ke Kepala UPT PMB
- Biaya: Rp 200.000

### Setelah 7 Hari Kuliah
- Tunggu semester berikutnya
- Surat ke Dekan c.q. Kaprodi
""",

    "pengunduran_diri": """
## PENGUNDURAN DIRI & PENGEMBALIAN DANA

### Sebelum Orientasi Mahasiswa Baru
- Dana pendidikan dikembalikan 25%
- Biaya pendaftaran TIDAK dikembalikan

### Setelah Kuliah Dimulai
- Seluruh dana TIDAK dikembalikan

### Force Majeure (sakit parah, bencana)
- Bisa dikembalikan dengan kebijakan pimpinan
- Buat surat permohonan dengan bukti

### Pindah ke PT Lain
- Dana pendidikan dikembalikan 25%
- Dana Atribut, Muawanah, Matrikulasi, Kemahasiswaan TIDAK dikembalikan
- Jaz dan buku bisa diambil
""",

    "transfer": """
## MAHASISWA TRANSFER

### Transfer dari PT Lain - Belum Lulus
1. Siapkan surat pindah bermeterai + KHS dari PT asal
2. Bawa ke Kaprodi UNSIQ untuk konversi nilai
3. Jika sepakat, berarti diterima
4. Daftar di portal PMB jalur RPL - Transfer Kredit
5. Upload dokumen, ujian, bayar semester 1
6. Tentukan matkul berdasarkan konversi
7. Sesuaikan biaya SKS di Bagian Registrasi

### Transfer dari PT Lain - Sudah Lulus
1. Siapkan ijazah + transkrip dari PT asal
2. Konversi nilai dengan Kaprodi UNSIQ
3. Daftar jalur RPL di portal PMB
4. Upload dokumen, ujian, bayar
5. Tentukan matkul dan sesuaikan biaya

### Catatan Mahasiswa Transfer
- Gunakan sistem RPL (Rekognisi Pembelajaran Lampau)
- Tidak semua matkul bisa dikonversi
- Durasi studi mungkin lebih pendek
""",

    "faq": """
## FAQ PENDAFTARAN

### Bisa daftar dua gelombang?
- Tidak. Hanya satu gelombang untuk satu prodi.

### Lupa nomor pendaftaran/PIN?
- Hubungi PMB: 0857 7504 7504
- Siapkan identitas untuk verifikasi

### Bisa ubah prodi setelah daftar?
- Sebelum ujian: Ubah sendiri (gratis)
- Setelah ujian: Hubungi PMB reset (gratis)
- Setelah reservasi NIM: Surat + Rp 200.000
- Setelah 7 hari kuliah: Tunggu semester depan

### Belum punya SKL?
- Pakai surat keterangan aktif dari sekolah
- Upload ijazah asli saat sudah keluar

### Tidak lulus ujian?
- Bisa ikut ujian di gelombang berikutnya
- Bayar biaya pendaftaran gelombang baru

### Setelah diterima, apa selanjutnya?
1. Bayar semester 1
2. Herregistrasi
3. Reservasi NIM
4. Ikuti orientasi
5. Lengkapi administrasi
""",

    "kontak": """
## KONTAK & INFORMASI

### Layanan PMB UNSIQ
- Telepon: (0286) 321873
- WhatsApp: 0857 7504 7504
- Telegram: 0857 7504 7504
- Email: humas@unsiq.ac.id
- Website: https://pmb.unsiq.ac.id/
- Jam: Senin-Jumat, 08:00-16:00 WIB

### Alamat
Universitas Sains Al-Qur'an (UNSIQ)
Jl. KH. Hasyim Asy'ari Km. 03, Kalibebar
Kec. Mojotengah, Kab. Wonosobo
Jawa Tengah 56351
"""
}

# =============================================================================
# 150 SCENARIOS (will generate 2x with different personas = 300 total)
# =============================================================================

SCENARIOS = [
    # === TIMELINE (15 scenarios) ===
    {"id": "TL01", "category": "timeline", "scenario": "User tanya jadwal pendaftaran Gelombang 1", "complexity": "direct"},
    {"id": "TL02", "category": "timeline", "scenario": "User tanya jadwal pendaftaran Gelombang 2", "complexity": "direct"},
    {"id": "TL03", "category": "timeline", "scenario": "User tanya jadwal pendaftaran Gelombang 3", "complexity": "direct"},
    {"id": "TL04", "category": "timeline", "scenario": "User tanya jadwal pendaftaran S2", "complexity": "direct"},
    {"id": "TL05", "category": "timeline", "scenario": "User tanya deadline ujian placement", "complexity": "direct"},
    {"id": "TL06", "category": "timeline", "scenario": "User tanya deadline bayar semester 1", "complexity": "direct"},
    {"id": "TL07", "category": "timeline", "scenario": "User tanya gelombang mana yang gratis", "complexity": "reasoning"},
    {"id": "TL08", "category": "timeline", "scenario": "User tanya perbedaan tiap gelombang", "complexity": "reasoning"},
    {"id": "TL09", "category": "timeline", "scenario": "User tanya keuntungan daftar Gelombang 1", "complexity": "reasoning"},
    {"id": "TL10", "category": "timeline", "scenario": "User telat daftar Gelombang 1 tanya opsi", "complexity": "edge_case"},
    {"id": "TL11", "category": "timeline", "scenario": "User bingung pilih gelombang mana", "complexity": "reasoning"},
    {"id": "TL12", "category": "timeline", "scenario": "User tanya kapan kuliah dimulai", "complexity": "direct"},
    {"id": "TL13", "category": "timeline", "scenario": "User tanya total durasi proses pendaftaran", "complexity": "reasoning"},
    {"id": "TL14", "category": "timeline", "scenario": "User tanya jadwal orientasi mahasiswa baru", "complexity": "direct"},
    {"id": "TL15", "category": "timeline", "scenario": "User tanya timeline lengkap dari daftar sampai jadi mahasiswa", "complexity": "reasoning"},

    # === DOKUMEN (20 scenarios) ===
    {"id": "DK01", "category": "dokumen", "scenario": "User tanya dokumen apa saja yang diperlukan", "complexity": "direct"},
    {"id": "DK02", "category": "dokumen", "scenario": "User tanya format file yang diterima", "complexity": "direct"},
    {"id": "DK03", "category": "dokumen", "scenario": "User tanya ukuran maksimal file upload", "complexity": "direct"},
    {"id": "DK04", "category": "dokumen", "scenario": "User belum punya ijazah tanya alternatif", "complexity": "edge_case"},
    {"id": "DK05", "category": "dokumen", "scenario": "User tanya format dan ukuran foto", "complexity": "direct"},
    {"id": "DK06", "category": "dokumen", "scenario": "User tanya background foto harus apa", "complexity": "direct"},
    {"id": "DK07", "category": "dokumen", "scenario": "User KK sudah kadaluarsa tanya solusi", "complexity": "edge_case"},
    {"id": "DK08", "category": "dokumen", "scenario": "User tanya surat kesehatan dari mana", "complexity": "direct"},
    {"id": "DK09", "category": "dokumen", "scenario": "User bukan FIKES tanya perlu surat sehat tidak", "complexity": "reasoning"},
    {"id": "DK10", "category": "dokumen", "scenario": "User lupa PIN/password tanya cara reset", "complexity": "edge_case"},
    {"id": "DK11", "category": "dokumen", "scenario": "User tanya apakah SKHUN wajib", "complexity": "direct"},
    {"id": "DK12", "category": "dokumen", "scenario": "User tanya dokumen tambahan untuk FIKES", "complexity": "reasoning"},
    {"id": "DK13", "category": "dokumen", "scenario": "User tanya cara bayar pendaftaran", "complexity": "direct"},
    {"id": "DK14", "category": "dokumen", "scenario": "User tanya referensi pembayaran yang benar", "complexity": "reasoning"},
    {"id": "DK15", "category": "dokumen", "scenario": "User sudah bayar tapi status belum berubah", "complexity": "edge_case"},
    {"id": "DK16", "category": "dokumen", "scenario": "User salah transfer nominal pembayaran", "complexity": "edge_case"},
    {"id": "DK17", "category": "dokumen", "scenario": "User tanya cara bayar via Bank Jateng", "complexity": "reasoning"},
    {"id": "DK18", "category": "dokumen", "scenario": "User tanya cara bayar via Finpay", "complexity": "reasoning"},
    {"id": "DK19", "category": "dokumen", "scenario": "User upload dokumen tapi ditolak", "complexity": "edge_case"},
    {"id": "DK20", "category": "dokumen", "scenario": "User tanya checklist dokumen lengkap", "complexity": "reasoning"},

    # === PROSES PENDAFTARAN (20 scenarios) ===
    {"id": "PR01", "category": "proses", "scenario": "User tanya langkah pertama pendaftaran", "complexity": "direct"},
    {"id": "PR02", "category": "proses", "scenario": "User tanya cara isi formulir online", "complexity": "reasoning"},
    {"id": "PR03", "category": "proses", "scenario": "User tanya cara login ke portal PMB", "complexity": "direct"},
    {"id": "PR04", "category": "proses", "scenario": "User tanya berapa lama proses verifikasi", "complexity": "direct"},
    {"id": "PR05", "category": "proses", "scenario": "User verifikasi ditolak tanya alasan", "complexity": "edge_case"},
    {"id": "PR06", "category": "proses", "scenario": "User tanya cara ikut ujian placement", "complexity": "reasoning"},
    {"id": "PR07", "category": "proses", "scenario": "User tanya durasi dan materi ujian placement", "complexity": "reasoning"},
    {"id": "PR08", "category": "proses", "scenario": "User gagal ujian tanya opsi selanjutnya", "complexity": "edge_case"},
    {"id": "PR09", "category": "proses", "scenario": "User tanya cara bayar semester 1", "complexity": "reasoning"},
    {"id": "PR10", "category": "proses", "scenario": "User tanya minimal bayar semester 1", "complexity": "direct"},
    {"id": "PR11", "category": "proses", "scenario": "User tanya apa itu reservasi NIM", "complexity": "direct"},
    {"id": "PR12", "category": "proses", "scenario": "User tanya cara reservasi NIM", "complexity": "reasoning"},
    {"id": "PR13", "category": "proses", "scenario": "User tanya apa yang dilakukan setelah diterima", "complexity": "reasoning"},
    {"id": "PR14", "category": "proses", "scenario": "User tanya apa itu herregistrasi", "complexity": "direct"},
    {"id": "PR15", "category": "proses", "scenario": "User bingung langkah selanjutnya", "complexity": "edge_case"},
    {"id": "PR16", "category": "proses", "scenario": "User tanya apakah bisa daftar offline", "complexity": "direct"},
    {"id": "PR17", "category": "proses", "scenario": "User dari luar kota tanya proses pendaftaran", "complexity": "reasoning"},
    {"id": "PR18", "category": "proses", "scenario": "User tanya kontak untuk konsultasi pendaftaran", "complexity": "direct"},
    {"id": "PR19", "category": "proses", "scenario": "User tanya proses untuk mahasiswa transfer", "complexity": "reasoning"},
    {"id": "PR20", "category": "proses", "scenario": "User tanya proses lengkap dari awal sampai akhir", "complexity": "reasoning"},

    # === PERSYARATAN KHUSUS (15 scenarios) ===
    {"id": "PK01", "category": "persyaratan_khusus", "scenario": "User tanya syarat tinggi badan FIKES", "complexity": "direct"},
    {"id": "PK02", "category": "persyaratan_khusus", "scenario": "User tinggi badan kurang dari syarat tanya solusi", "complexity": "edge_case"},
    {"id": "PK03", "category": "persyaratan_khusus", "scenario": "User tanya syarat S1 Keperawatan kelas karyawan", "complexity": "direct"},
    {"id": "PK04", "category": "persyaratan_khusus", "scenario": "User lulusan SMA mau masuk S1 Keperawatan karyawan", "complexity": "edge_case"},
    {"id": "PK05", "category": "persyaratan_khusus", "scenario": "User tanya persyaratan khusus per fakultas", "complexity": "reasoning"},
    {"id": "PK06", "category": "persyaratan_khusus", "scenario": "User tanya tes buta warna untuk siapa", "complexity": "direct"},
    {"id": "PK07", "category": "persyaratan_khusus", "scenario": "User tanya syarat usia maksimal", "complexity": "direct"},
    {"id": "PK08", "category": "persyaratan_khusus", "scenario": "User tanya syarat khusus untuk S2", "complexity": "reasoning"},
    {"id": "PK09", "category": "persyaratan_khusus", "scenario": "User tanya syarat akademik minimal", "complexity": "direct"},
    {"id": "PK10", "category": "persyaratan_khusus", "scenario": "User memiliki disabilitas tanya bisa daftar tidak", "complexity": "edge_case"},
    {"id": "PK11", "category": "persyaratan_khusus", "scenario": "User tanya syarat untuk jurusan kesehatan", "complexity": "reasoning"},
    {"id": "PK12", "category": "persyaratan_khusus", "scenario": "User tanya siapa yang bisa masuk FIKES", "complexity": "reasoning"},
    {"id": "PK13", "category": "persyaratan_khusus", "scenario": "User tanya persyaratan bahasa arab untuk prodi tertentu", "complexity": "direct"},
    {"id": "PK14", "category": "persyaratan_khusus", "scenario": "User tanya syarat hafalan quran untuk beasiswa", "complexity": "reasoning"},
    {"id": "PK15", "category": "persyaratan_khusus", "scenario": "User dari luar negeri tanya syarat khusus", "complexity": "edge_case"},

    # === KETENTUAN & KONSEKUENSI (15 scenarios) ===
    {"id": "KT01", "category": "ketentuan", "scenario": "User tanya apakah uang pendaftaran bisa refund", "complexity": "direct"},
    {"id": "KT02", "category": "ketentuan", "scenario": "User tanya konsekuensi tidak ikut ujian", "complexity": "direct"},
    {"id": "KT03", "category": "ketentuan", "scenario": "User tanya konsekuensi tidak herregistrasi", "complexity": "direct"},
    {"id": "KT04", "category": "ketentuan", "scenario": "User tanya konsekuensi tidak kuliah semester 1", "complexity": "direct"},
    {"id": "KT05", "category": "ketentuan", "scenario": "User tanya apa saja yang dianggap mengundurkan diri", "complexity": "reasoning"},
    {"id": "KT06", "category": "ketentuan", "scenario": "User tidak bisa bayar tepat waktu tanya solusi", "complexity": "edge_case"},
    {"id": "KT07", "category": "ketentuan", "scenario": "User tanya apakah bisa perpanjang deadline", "complexity": "edge_case"},
    {"id": "KT08", "category": "ketentuan", "scenario": "User tanya ketentuan pembayaran semester", "complexity": "reasoning"},
    {"id": "KT09", "category": "ketentuan", "scenario": "User tanya aturan cuti kuliah", "complexity": "direct"},
    {"id": "KT10", "category": "ketentuan", "scenario": "User tanya sanksi jika melanggar aturan", "complexity": "reasoning"},
    {"id": "KT11", "category": "ketentuan", "scenario": "User tanya hak dan kewajiban mahasiswa baru", "complexity": "reasoning"},
    {"id": "KT12", "category": "ketentuan", "scenario": "User tanya kebijakan kehadiran", "complexity": "direct"},
    {"id": "KT13", "category": "ketentuan", "scenario": "User tanya ketentuan akademik semester 1", "complexity": "reasoning"},
    {"id": "KT14", "category": "ketentuan", "scenario": "User tanya berapa minimal SKS per semester", "complexity": "direct"},
    {"id": "KT15", "category": "ketentuan", "scenario": "User tanya ketentuan untuk mahasiswa aktif", "complexity": "reasoning"},

    # === PINDAH PRODI (15 scenarios) ===
    {"id": "PP01", "category": "pindah_prodi", "scenario": "User tanya cara pindah prodi sebelum ujian", "complexity": "direct"},
    {"id": "PP02", "category": "pindah_prodi", "scenario": "User tanya cara pindah prodi setelah ujian", "complexity": "reasoning"},
    {"id": "PP03", "category": "pindah_prodi", "scenario": "User tanya cara pindah prodi setelah dapat NIM", "complexity": "reasoning"},
    {"id": "PP04", "category": "pindah_prodi", "scenario": "User tanya biaya pindah prodi", "complexity": "direct"},
    {"id": "PP05", "category": "pindah_prodi", "scenario": "User tanya deadline pindah prodi", "complexity": "direct"},
    {"id": "PP06", "category": "pindah_prodi", "scenario": "User sudah kuliah mau pindah prodi", "complexity": "edge_case"},
    {"id": "PP07", "category": "pindah_prodi", "scenario": "User tanya prosedur reset ujian placement", "complexity": "reasoning"},
    {"id": "PP08", "category": "pindah_prodi", "scenario": "User tanya surat apa untuk pindah prodi", "complexity": "direct"},
    {"id": "PP09", "category": "pindah_prodi", "scenario": "User tanya ke siapa mengajukan pindah prodi", "complexity": "direct"},
    {"id": "PP10", "category": "pindah_prodi", "scenario": "User pindah dari FIKES ke non-FIKES", "complexity": "edge_case"},
    {"id": "PP11", "category": "pindah_prodi", "scenario": "User pindah dari non-FIKES ke FIKES", "complexity": "edge_case"},
    {"id": "PP12", "category": "pindah_prodi", "scenario": "User tanya apakah nilai bisa dikonversi saat pindah prodi", "complexity": "reasoning"},
    {"id": "PP13", "category": "pindah_prodi", "scenario": "User tanya kapan waktu terbaik pindah prodi", "complexity": "reasoning"},
    {"id": "PP14", "category": "pindah_prodi", "scenario": "User baru daftar tapi salah pilih prodi", "complexity": "edge_case"},
    {"id": "PP15", "category": "pindah_prodi", "scenario": "User tanya proses lengkap pindah prodi", "complexity": "reasoning"},

    # === PENGUNDURAN DIRI & REFUND (15 scenarios) ===
    {"id": "PD01", "category": "pengunduran_diri", "scenario": "User tanya cara mengundurkan diri", "complexity": "direct"},
    {"id": "PD02", "category": "pengunduran_diri", "scenario": "User tanya berapa persen refund sebelum orientasi", "complexity": "direct"},
    {"id": "PD03", "category": "pengunduran_diri", "scenario": "User tanya apakah ada refund setelah kuliah dimulai", "complexity": "direct"},
    {"id": "PD04", "category": "pengunduran_diri", "scenario": "User tanya prosedur pengunduran diri", "complexity": "reasoning"},
    {"id": "PD05", "category": "pengunduran_diri", "scenario": "User tanya dokumen untuk mengundurkan diri", "complexity": "direct"},
    {"id": "PD06", "category": "pengunduran_diri", "scenario": "User sakit parah tanya kebijakan force majeure", "complexity": "edge_case"},
    {"id": "PD07", "category": "pengunduran_diri", "scenario": "User mau pindah ke PT lain tanya refund", "complexity": "reasoning"},
    {"id": "PD08", "category": "pengunduran_diri", "scenario": "User tanya dana apa saja yang tidak dikembalikan", "complexity": "direct"},
    {"id": "PD09", "category": "pengunduran_diri", "scenario": "User tanya apakah jaz dan buku bisa diambil", "complexity": "direct"},
    {"id": "PD10", "category": "pengunduran_diri", "scenario": "User tanya proses pengajuan refund", "complexity": "reasoning"},
    {"id": "PD11", "category": "pengunduran_diri", "scenario": "User tanya berapa lama proses refund", "complexity": "direct"},
    {"id": "PD12", "category": "pengunduran_diri", "scenario": "User tanya ke siapa mengajukan pengunduran diri", "complexity": "direct"},
    {"id": "PD13", "category": "pengunduran_diri", "scenario": "User keluarga meninggal tanya kebijakan khusus", "complexity": "edge_case"},
    {"id": "PD14", "category": "pengunduran_diri", "scenario": "User tanya contoh perhitungan refund", "complexity": "reasoning"},
    {"id": "PD15", "category": "pengunduran_diri", "scenario": "User diterima SNBP/SNBT tanya cara mundur dari UNSIQ", "complexity": "edge_case"},

    # === MAHASISWA TRANSFER (15 scenarios) ===
    {"id": "TR01", "category": "transfer", "scenario": "User tanya cara daftar sebagai mahasiswa transfer", "complexity": "reasoning"},
    {"id": "TR02", "category": "transfer", "scenario": "User tanya dokumen untuk transfer", "complexity": "direct"},
    {"id": "TR03", "category": "transfer", "scenario": "User tanya apa itu jalur RPL", "complexity": "direct"},
    {"id": "TR04", "category": "transfer", "scenario": "User tanya cara konversi nilai dari PT asal", "complexity": "reasoning"},
    {"id": "TR05", "category": "transfer", "scenario": "User tanya apakah semua matkul bisa dikonversi", "complexity": "direct"},
    {"id": "TR06", "category": "transfer", "scenario": "User tanya siapa yang menentukan konversi", "complexity": "direct"},
    {"id": "TR07", "category": "transfer", "scenario": "User transfer belum lulus dari PT asal", "complexity": "reasoning"},
    {"id": "TR08", "category": "transfer", "scenario": "User transfer sudah lulus dari PT asal", "complexity": "reasoning"},
    {"id": "TR09", "category": "transfer", "scenario": "User tanya biaya untuk mahasiswa transfer", "complexity": "direct"},
    {"id": "TR10", "category": "transfer", "scenario": "User tanya durasi studi untuk transfer", "complexity": "reasoning"},
    {"id": "TR11", "category": "transfer", "scenario": "User tanya cara sesuaikan biaya SKS", "complexity": "reasoning"},
    {"id": "TR12", "category": "transfer", "scenario": "User tanya proses lengkap transfer dari PT lain", "complexity": "reasoning"},
    {"id": "TR13", "category": "transfer", "scenario": "User transfer dari prodi berbeda", "complexity": "edge_case"},
    {"id": "TR14", "category": "transfer", "scenario": "User tanya ke mana menghubungi untuk transfer", "complexity": "direct"},
    {"id": "TR15", "category": "transfer", "scenario": "User tanya apakah harus ujian ulang untuk transfer", "complexity": "direct"},

    # === FAQ & KONTAK (20 scenarios) ===
    {"id": "FQ01", "category": "faq", "scenario": "User tanya apakah bisa daftar dua gelombang sekaligus", "complexity": "direct"},
    {"id": "FQ02", "category": "faq", "scenario": "User lupa nomor pendaftaran", "complexity": "edge_case"},
    {"id": "FQ03", "category": "faq", "scenario": "User tanya cara reset password portal", "complexity": "direct"},
    {"id": "FQ04", "category": "faq", "scenario": "User tanya kontak PMB UNSIQ", "complexity": "direct"},
    {"id": "FQ05", "category": "faq", "scenario": "User tanya jam layanan PMB", "complexity": "direct"},
    {"id": "FQ06", "category": "faq", "scenario": "User tanya nomor WhatsApp PMB", "complexity": "direct"},
    {"id": "FQ07", "category": "faq", "scenario": "User tanya alamat kampus UNSIQ", "complexity": "direct"},
    {"id": "FQ08", "category": "faq", "scenario": "User tanya email resmi untuk kontak", "complexity": "direct"},
    {"id": "FQ09", "category": "faq", "scenario": "User tidak lulus ujian tanya opsi", "complexity": "edge_case"},
    {"id": "FQ10", "category": "faq", "scenario": "User tanya apa yang harus dilakukan setelah diterima", "complexity": "reasoning"},
    {"id": "FQ11", "category": "faq", "scenario": "User tanya apakah ada beasiswa saat pendaftaran", "complexity": "direct"},
    {"id": "FQ12", "category": "faq", "scenario": "User tanya website resmi PMB", "complexity": "direct"},
    {"id": "FQ13", "category": "faq", "scenario": "User tanya apakah bisa konsultasi langsung ke kampus", "complexity": "direct"},
    {"id": "FQ14", "category": "faq", "scenario": "User tanya cara komplain jika ada masalah", "complexity": "reasoning"},
    {"id": "FQ15", "category": "faq", "scenario": "User tanya apakah ada grup WA untuk calon mahasiswa", "complexity": "direct"},
    {"id": "FQ16", "category": "faq", "scenario": "User di luar jam kerja butuh bantuan", "complexity": "edge_case"},
    {"id": "FQ17", "category": "faq", "scenario": "User tanya apakah bisa tanya lewat DM Instagram", "complexity": "direct"},
    {"id": "FQ18", "category": "faq", "scenario": "User tanya ada tidak open house atau sosialisasi", "complexity": "direct"},
    {"id": "FQ19", "category": "faq", "scenario": "User tanya tips agar lolos seleksi", "complexity": "reasoning"},
    {"id": "FQ20", "category": "faq", "scenario": "User tanya ringkasan syarat ketentuan pendaftaran", "complexity": "reasoning"},
]

print(f"Total scenarios: {len(SCENARIOS)}")

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

SYSTEM_PROMPT = """You are an expert Synthetic Data Generator for UNSIQ (Universitas Sains Al-Qur'an).
Generate HIGH-QUALITY, REALISTIC MULTI-TURN conversations for training a Customer Service AI.

STRICT RULES:
1. **CONTEXT**: Use ONLY facts from the provided context. Do NOT hallucinate.
2. **USER STYLE**: Based on persona - can be casual, formal, or mixed.
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
✅ "Baik, berikut informasi syarat pendaftaran: 1. Formulir online 2. Scan ijazah 3. Kartu Keluarga"
"""

USER_PROMPT_TEMPLATE = """
CONTEXT:
{context}

SCENARIO: {scenario}
PERSONA: {persona_name} - {persona_desc}
COMPLEXITY: {complexity}

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
    print("SNK (SYARAT & KETENTUAN) DATASET GENERATOR")
    print(f"Target: {len(SCENARIOS)} scenarios x 2 personas = {len(SCENARIOS)*2} conversations")
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
    output_file = os.path.join(output_dir, "multiturn_snk_300.json")
    
    # Generate 2 variations per scenario with different personas
    generated_data = []
    batch_size = 10
    personas_list = list(PERSONAS.keys())
    
    # Create expanded scenarios (each scenario x 2 different personas)
    expanded_scenarios = []
    for scenario in SCENARIOS:
        # Pick 2 different personas for each scenario
        selected_personas = random.sample(personas_list, min(2, len(personas_list)))
        for i, persona in enumerate(selected_personas):
            expanded_scenarios.append({
                **scenario,
                "variation": i + 1,
                "persona": persona
            })
    
    random.shuffle(expanded_scenarios)  # Shuffle for diversity
    total = len(expanded_scenarios)
    
    pbar = tqdm(total=total, desc="Generating conversations", unit="conv")
    
    for batch_start in range(0, total, batch_size):
        batch_scenarios = expanded_scenarios[batch_start:batch_start+batch_size]
        
        # Build prompts
        prompts = []
        for sc in batch_scenarios:
            persona_key = sc["persona"]
            persona_desc = PERSONAS[persona_key]
            
            # Get category-specific context
            context = CONTEXT_SNK.get(sc["category"], CONTEXT_SNK["proses"])
            
            prompt = USER_PROMPT_TEMPLATE.format(
                context=context,
                scenario=sc["scenario"],
                persona_name=persona_key,
                persona_desc=persona_desc,
                complexity=sc["complexity"]
            )
            
            formatted = f"<bos><start_of_turn>user\n{SYSTEM_PROMPT}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            prompts.append(formatted)
        
        # Generate batch
        outputs = engine.generate_batch(prompts, max_tokens=1024, temperature=0.7)
        
        # Parse results
        for i, response in enumerate(outputs):
            sc = batch_scenarios[i]
            conversation = generator._parse_response(response)
            
            if conversation:
                item = {
                    "id": f"{sc['id']}_v{sc['variation']}",
                    "instruction": f"Multi-turn conversation about UNSIQ SNK - {sc['category']}",
                    "input": "",
                    "output": json.dumps(conversation, ensure_ascii=False),
                    "text": "",
                    "category": "snk",
                    "subcategory": sc["category"],
                    "scenario": sc["scenario"],
                    "complexity": sc["complexity"],
                    "persona": sc["persona"],
                    "source": "synthetic_snk_v1"
                }
                generated_data.append(item)
                pbar.update(1)
        
        # Checkpoint every batch
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(generated_data, f, ensure_ascii=False, indent=2)
    
    pbar.close()
    
    # Final save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nDONE! Generated {len(generated_data)} conversations")
    print(f"Saved to: {output_file}")
    
    # Print stats
    print("\n" + "="*60)
    print("STATISTICS:")
    print("="*60)
    from collections import Counter
    cat_counts = Counter(item["subcategory"] for item in generated_data)
    persona_counts = Counter(item["persona"] for item in generated_data)
    
    print("\nBy Category:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")
    
    print("\nBy Persona:")
    for persona, count in sorted(persona_counts.items()):
        print(f"  {persona}: {count}")


if __name__ == "__main__":
    main()

