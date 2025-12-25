#!/usr/bin/env python3
"""
High-Quality Alur Pendaftaran Multi-turn Dataset Generator
Generates diverse, non-repetitive conversations with FORMAL response style.
Uses SCENARIO-SPECIFIC context to stay within model token limits.

Usage: python generate_alur_pendaftaran.py
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
# SCENARIO-SPECIFIC CONTEXT MAPPING
# Each scenario gets only the relevant excerpt from the RAG document
# =============================================================================

CONTEXT_BY_STAGE = {
    "website": """
## TAHAP 1: AKSES WEBSITE PMB

### URL Resmi:
- Website utama: pmb.unsiq.ac.id
- Alternatif: pmb.unsiq.ac.id/2025

### Troubleshooting:
1. Clear cache browser: Ctrl+Shift+Delete
2. Coba browser berbeda (Chrome, Firefox, Safari)
3. Cek koneksi internet, restart modem
4. Hubungi PMB: 📱 0857 7504 7504 atau (0286) 321873

### Yang Akan Anda Lihat:
- 📝 Formulir Pendaftaran
- 🔑 Login
- 📢 Pengumuman
- ❓ FAQ/Bantuan
""",

    "formulir": """
## TAHAP 2: ISI FORMULIR PENDAFTARAN

### Durasi: 10-15 Menit

### Pilih Jenjang:
- D3 (Diploma Tiga) - 3 tahun
- S1 (Sarjana) - 4 tahun  
- S2 (Pascasarjana) - 2-3 tahun

### Pilih Kelas:
- Reguler - Untuk siswa baru lulusan SMA/SMK
- Ekstension - Untuk yang sudah bekerja

### Data yang Diisi:
- Nama Lengkap (sesuai ijazah)
- Email (yang masih aktif)
- Nomor Handphone
- Alamat lengkap

### PENTING - Setelah Daftar:
Sistem menampilkan Nomor Pendaftaran & PIN/Password.
WAJIB screenshot atau catat di 3 tempat berbeda!

### Lupa Nomor/PIN:
1. Cek email
2. Klik "Lupa Password?" di halaman login
3. Hubungi PMB: 📱 0857 7504 7504
""",

    "pembayaran_pendaftaran": """
## TAHAP 3: BAYAR BIAYA PENDAFTARAN

### Besaran Biaya:
| Gelombang | D3 & S1 | S2 |
|-----------|---------|-----|
| Gelombang 1 | GRATIS | GRATIS |
| Gelombang 2 | Rp 250.000 | Rp 300.000 |
| Gelombang 3 | Rp 250.000 | Rp 300.000 |

### Metode Pembayaran:
- Bank Jateng (Aplikasi/ATM)
- Finpay (Mobile payment)

### Cara Bayar:
1. Buka aplikasi Bank Jateng/Finpay
2. Pilih menu Transfer/Pembayaran
3. MASUKKAN NOMOR PENDAFTARAN sebagai referensi (BUKAN nomor rekening!)
4. Masukkan nominal sesuai gelombang
5. Konfirmasi & masukkan PIN
6. Screenshot bukti transfer

### Jika Pembayaran Tidak Terdeteksi:
- Tunggu 1-2 jam untuk sinkronisasi
- Pastikan pakai nomor pendaftaran (bukan rekening)
- Jika masih gagal, hubungi PMB dengan bukti transfer
""",

    "dokumen": """
## TAHAP 4: UPLOAD DOKUMEN

### Durasi: 20-30 Menit

### Dokumen yang Diperlukan (Format JPG):
| No | Dokumen | Spesifikasi |
|----|---------|-------------|
| 1 | Ijazah/SKL | Scan asli. Lulusan 2025 bisa pakai surat keterangan aktif |
| 2 | Kartu Keluarga | KK yang masih berlaku, jelas terbaca |
| 3 | Foto Formal | 250x350 pixel, background putih, formal |
| 4 | Surat Sehat | KHUSUS Fakultas Kesehatan - dari RS/Puskesmas, tes buta warna (laki-laki) |

### Cara Login:
1. Buka pmb.unsiq.ac.id
2. Klik "Login"
3. Masukkan Nomor Pendaftaran & PIN
4. Klik menu "Biodata" atau "Dokumen"

### Verifikasi:
- Maksimal 2x24 jam
- Jika dokumen kurang/tidak jelas: ada notifikasi upload ulang
- Status berubah "TERVERIFIKASI" = Menu Ujian aktif
""",

    "placement_test": """
## TAHAP 5: UJIAN PLACEMENT TEST

### Durasi: 60-90 menit
### Jadwal: Online, kapan saja sebelum deadline

### Syarat:
- Status dokumen: TERVERIFIKASI
- Menu "Ujian" sudah aktif

### Jika Menu Ujian Tidak Muncul:
1. Pastikan verifikasi dokumen sudah selesai
2. Cek email untuk notifikasi
3. Hubungi PMB jika lebih dari 2x24 jam

### Tips:
- Koneksi internet stabil
- Browser terbaru
- Jangan refresh saat ujian berlangsung
""",

    "pengumuman": """
## TAHAP 6: PENGUMUMAN HASIL

### Cara Cek:
1. Login ke portal PMB
2. Klik menu "Status Pendaftar" atau "Pengumuman"
3. Lihat status penerimaan

### Status:
- DITERIMA: Lanjut ke pembayaran semester 1
- WAITING: Menunggu proses verifikasi
- DITOLAK: Hubungi PMB untuk opsi selanjutnya
""",

    "registrasi": """
## TAHAP 7: BAYAR BIAYA SEMESTER 1

### Biaya Angsuran Pertama:
- D3/S1: Rp 745.000
- S2: Rp 1.100.000

### Cara Bayar:
Sama dengan pembayaran pendaftaran (Bank Jateng/Finpay)
- Gunakan NOMOR PENDAFTARAN sebagai referensi
- Pembayaran masuk dalam 1-2 jam

### Jika Status Tidak Update:
- Tunggu maksimal 2 jam
- Hubungi PMB dengan bukti transfer

### Rincian Biaya Lengkap:
Lihat dokumen "Komponen Biaya Pendidikan" untuk detail cicilan
""",

    "nim": """
## TAHAP 8: RESERVASI NIM

### Syarat: Pembayaran semester 1 berhasil

### Cara Reservasi:
1. Login ke portal PMB
2. Cek status pembayaran: LUNAS/TERVERIFIKASI
3. Klik tombol "Reservasi NIM"
4. Konfirmasi data
5. Tunggu verifikasi: 1-3 hari kerja

### Setelah Dapat NIM:
1. Ikuti orientasi mahasiswa
2. Ambil Kartu Mahasiswa (KTM)
3. Daftar matakuliah di portal akademik
4. Bayar cicilan sesuai jadwal
5. Ikuti kuliah sesuai jadwal

### Jika Tombol Reservasi Tidak Aktif:
- Pastikan pembayaran sudah terverifikasi
- Hubungi PMB jika lebih dari 2 jam
""",

    "general": """
## ALUR PENDAFTARAN LENGKAP

### 8 Tahap Pendaftaran:
1. Akses website pmb.unsiq.ac.id
2. Isi formulir → dapat Nomor Pendaftaran & PIN
3. Bayar biaya pendaftaran (Gelombang 1 GRATIS)
4. Login & upload dokumen (Ijazah, KK, Foto, Surat Sehat)
5. Ujian Placement Test (60-90 menit, online)
6. Cek pengumuman hasil
7. Bayar semester 1 (D3/S1: Rp745rb, S2: Rp1.1jt)
8. Reservasi NIM → RESMI MAHASISWA UNSIQ

### Pindah Prodi:
- Sebelum ujian: Ubah sendiri di menu Biodata (gratis)
- Setelah ujian: Hubungi PMB untuk reset (gratis)
- Setelah dapat NIM: Surat ke Kepala PMB + Rp 200.000 (max 7 hari)

### Kontak PMB:
- Telepon: (0286) 321873
- WhatsApp: 0857 7504 7504
- Email: humas@unsiq.ac.id
- Jam Layanan: Senin-Jumat, 08:00-16:00 WIB

### Proses Online:
Semua tahap bisa dilakukan online dari mana saja.
""",

    "jadwal": """
## JADWAL GELOMBANG PMB UNSIQ

| Gelombang | Periode Pendaftaran | Biaya Pendaftaran | Jadwal Ujian | Bayar Angsuran 1 | Perkuliahan Dimulai |
|-----------|---------------------|-------------------|--------------|------------------|---------------------|
| Gelombang 1 | Nov-Des 2025 | Rp 0 (GRATIS) | Des 2025 | Des 2025 | Secepatnya |
| Gelombang 2 | Jan-Apr 2026 | Rp 250.000 | Mei 2026 | 15 Mei 2026 | Juni 2026 |
| Gelombang 3 | Mei-Jul 2026 | Rp 300.000 | Agustus 2026 | 15 Agustus 2026 | September 2026 |

### Keuntungan Gelombang 1:
- Biaya pendaftaran GRATIS
- Bisa mulai kuliah lebih cepat
- Kuota terbanyak

### Tips Memilih Gelombang:
- Sudah siap dokumen? Pilih Gelombang 1
- Belum lulus SMA? Tunggu Gelombang 2/3
"""
}

# =============================================================================
# SCENARIOS (ALL ANSWERABLE FROM RAG)
# =============================================================================

SCENARIOS = [
    # ==========================================================================
    # TAHAP 1: WEBSITE (15 skenario)
    # ==========================================================================
    {"id": "W01", "stage": "website", "scenario": "User tidak bisa akses website PMB, minta troubleshooting", "complexity": "edge_case"},
    {"id": "W02", "stage": "website", "scenario": "User tanya URL resmi PMB UNSIQ", "complexity": "direct"},
    {"id": "W03", "stage": "website", "scenario": "User tanya apakah ada website alternatif PMB", "complexity": "direct"},
    {"id": "W04", "stage": "website", "scenario": "User website loading lama, minta solusi", "complexity": "edge_case"},
    {"id": "W05", "stage": "website", "scenario": "User tanya browser apa yang direkomendasikan", "complexity": "reasoning"},
    {"id": "W06", "stage": "website", "scenario": "User website error 404, minta bantuan", "complexity": "edge_case"},
    {"id": "W07", "stage": "website", "scenario": "User tanya menu apa saja yang ada di website PMB", "complexity": "direct"},
    {"id": "W08", "stage": "website", "scenario": "User akses dari HP, tanya apakah bisa daftar lewat mobile", "complexity": "reasoning"},
    {"id": "W09", "stage": "website", "scenario": "User tanya cara clear cache browser", "complexity": "direct"},
    {"id": "W10", "stage": "website", "scenario": "User website blank putih, minta troubleshooting", "complexity": "edge_case"},
    {"id": "W11", "stage": "website", "scenario": "User tanya apakah pendaftaran bisa offline", "complexity": "reasoning"},
    {"id": "W12", "stage": "website", "scenario": "User tanya jam berapa website maintenance", "complexity": "direct"},
    {"id": "W13", "stage": "website", "scenario": "User dapat pesan 'server busy', minta solusi", "complexity": "edge_case"},
    {"id": "W14", "stage": "website", "scenario": "User tanya apakah perlu akun untuk akses website", "complexity": "direct"},
    {"id": "W15", "stage": "website", "scenario": "User dari luar negeri, tanya apakah bisa akses website", "complexity": "reasoning"},
    
    # ==========================================================================
    # TAHAP 2: FORMULIR (20 skenario)
    # ==========================================================================
    {"id": "F01", "stage": "formulir", "scenario": "User bingung memilih antara kelas Reguler vs Ekstension", "complexity": "reasoning"},
    {"id": "F02", "stage": "formulir", "scenario": "User tanya perbedaan jenjang D3, S1, S2", "complexity": "direct"},
    {"id": "F03", "stage": "formulir", "scenario": "User lupa nomor pendaftaran dan PIN setelah daftar", "complexity": "edge_case"},
    {"id": "F04", "stage": "formulir", "scenario": "User salah ketik nama saat pendaftaran", "complexity": "edge_case"},
    {"id": "F05", "stage": "formulir", "scenario": "User tanya data apa saja yang perlu diisi di formulir", "complexity": "direct"},
    {"id": "F06", "stage": "formulir", "scenario": "User tanya pentingnya mencatat nomor pendaftaran", "complexity": "reasoning"},
    {"id": "F07", "stage": "formulir", "scenario": "User tanya berapa lama waktu pengisian formulir", "complexity": "direct"},
    {"id": "F08", "stage": "formulir", "scenario": "User salah ketik email saat pendaftaran", "complexity": "edge_case"},
    {"id": "F09", "stage": "formulir", "scenario": "User tanya apakah bisa edit biodata setelah submit", "complexity": "reasoning"},
    {"id": "F10", "stage": "formulir", "scenario": "User mau daftar 2 prodi sekaligus", "complexity": "edge_case"},
    {"id": "F11", "stage": "formulir", "scenario": "User tanya format penulisan nama yang benar", "complexity": "direct"},
    {"id": "F12", "stage": "formulir", "scenario": "User tanya apakah harus pakai email aktif", "complexity": "reasoning"},
    {"id": "F13", "stage": "formulir", "scenario": "User mau ganti nomor HP setelah daftar", "complexity": "edge_case"},
    {"id": "F14", "stage": "formulir", "scenario": "User tanya apakah alamat harus lengkap", "complexity": "direct"},
    {"id": "F15", "stage": "formulir", "scenario": "User tanya siapa yang cocok untuk kelas Ekstension", "complexity": "reasoning"},
    {"id": "F16", "stage": "formulir", "scenario": "User belum punya KTP, tanya apakah bisa daftar", "complexity": "edge_case"},
    {"id": "F17", "stage": "formulir", "scenario": "User tanya prodi apa saja yang tersedia", "complexity": "direct"},
    {"id": "F18", "stage": "formulir", "scenario": "User bingung pilih prodi, minta rekomendasi", "complexity": "reasoning"},
    {"id": "F19", "stage": "formulir", "scenario": "User mau daftar untuk anak, tanya prosedur", "complexity": "edge_case"},
    {"id": "F20", "stage": "formulir", "scenario": "User tanya apakah bisa simpan draft formulir", "complexity": "direct"},
    
    # ==========================================================================
    # TAHAP 3: PEMBAYARAN PENDAFTARAN (20 skenario)
    # ==========================================================================
    {"id": "P01", "stage": "pembayaran_pendaftaran", "scenario": "User tanya biaya pendaftaran Gelombang 1", "complexity": "direct"},
    {"id": "P02", "stage": "pembayaran_pendaftaran", "scenario": "User tanya cara bayar lewat Bank Jateng", "complexity": "reasoning"},
    {"id": "P03", "stage": "pembayaran_pendaftaran", "scenario": "User transfer pakai nomor rekening bukan nomor pendaftaran", "complexity": "edge_case"},
    {"id": "P04", "stage": "pembayaran_pendaftaran", "scenario": "User tanya metode pembayaran yang tersedia", "complexity": "direct"},
    {"id": "P05", "stage": "pembayaran_pendaftaran", "scenario": "User sudah bayar tapi status belum berubah", "complexity": "edge_case"},
    {"id": "P06", "stage": "pembayaran_pendaftaran", "scenario": "User tanya perbedaan biaya D3/S1 vs S2", "complexity": "direct"},
    {"id": "P07", "stage": "pembayaran_pendaftaran", "scenario": "User tanya referensi pembayaran yang benar", "complexity": "reasoning"},
    {"id": "P08", "stage": "pembayaran_pendaftaran", "scenario": "User tanya biaya pendaftaran Gelombang 2", "complexity": "direct"},
    {"id": "P09", "stage": "pembayaran_pendaftaran", "scenario": "User tanya biaya pendaftaran Gelombang 3", "complexity": "direct"},
    {"id": "P10", "stage": "pembayaran_pendaftaran", "scenario": "User tanya cara bayar lewat Finpay", "complexity": "reasoning"},
    {"id": "P11", "stage": "pembayaran_pendaftaran", "scenario": "User transfer nominal kurang dari seharusnya", "complexity": "edge_case"},
    {"id": "P12", "stage": "pembayaran_pendaftaran", "scenario": "User tanya apakah bisa bayar via transfer antar bank", "complexity": "reasoning"},
    {"id": "P13", "stage": "pembayaran_pendaftaran", "scenario": "User bukti transfer hilang, minta solusi", "complexity": "edge_case"},
    {"id": "P14", "stage": "pembayaran_pendaftaran", "scenario": "User tanya deadline pembayaran pendaftaran", "complexity": "direct"},
    {"id": "P15", "stage": "pembayaran_pendaftaran", "scenario": "User mau bayar pakai QRIS", "complexity": "reasoning"},
    {"id": "P16", "stage": "pembayaran_pendaftaran", "scenario": "User double transfer, minta refund", "complexity": "edge_case"},
    {"id": "P17", "stage": "pembayaran_pendaftaran", "scenario": "User tanya apakah biaya bisa dicicil", "complexity": "reasoning"},
    {"id": "P18", "stage": "pembayaran_pendaftaran", "scenario": "User tanya langkah setelah pembayaran berhasil", "complexity": "direct"},
    {"id": "P19", "stage": "pembayaran_pendaftaran", "scenario": "User dari luar kota, tanya cara bayar", "complexity": "reasoning"},
    {"id": "P20", "stage": "pembayaran_pendaftaran", "scenario": "User pembayaran gagal, saldo terpotong", "complexity": "edge_case"},
    
    # ==========================================================================
    # TAHAP 4: DOKUMEN (20 skenario)
    # ==========================================================================
    {"id": "D01", "stage": "dokumen", "scenario": "User tanya dokumen yang perlu diupload", "complexity": "direct"},
    {"id": "D02", "stage": "dokumen", "scenario": "User belum punya ijazah, tanya pengganti", "complexity": "reasoning"},
    {"id": "D03", "stage": "dokumen", "scenario": "User tanya spesifikasi foto formal", "complexity": "direct"},
    {"id": "D04", "stage": "dokumen", "scenario": "User tanya lama verifikasi dokumen", "complexity": "direct"},
    {"id": "D05", "stage": "dokumen", "scenario": "User dapat notifikasi dokumen ditolak", "complexity": "edge_case"},
    {"id": "D06", "stage": "dokumen", "scenario": "User Fakultas Kesehatan tanya syarat surat sehat", "complexity": "reasoning"},
    {"id": "D07", "stage": "dokumen", "scenario": "User tanya format file yang diterima", "complexity": "direct"},
    {"id": "D08", "stage": "dokumen", "scenario": "User tanya cara login ke portal", "complexity": "direct"},
    {"id": "D09", "stage": "dokumen", "scenario": "User ijazah blur, tanya apakah diterima", "complexity": "edge_case"},
    {"id": "D10", "stage": "dokumen", "scenario": "User tanya ukuran maksimal file upload", "complexity": "direct"},
    {"id": "D11", "stage": "dokumen", "scenario": "User foto pakai background biru, tanya apakah valid", "complexity": "reasoning"},
    {"id": "D12", "stage": "dokumen", "scenario": "User KK sudah tidak berlaku, tanya solusi", "complexity": "edge_case"},
    {"id": "D13", "stage": "dokumen", "scenario": "User tanya apakah bisa upload dokumen dalam PDF", "complexity": "reasoning"},
    {"id": "D14", "stage": "dokumen", "scenario": "User gagal upload, muncul error", "complexity": "edge_case"},
    {"id": "D15", "stage": "dokumen", "scenario": "User tanya cara resize foto", "complexity": "direct"},
    {"id": "D16", "stage": "dokumen", "scenario": "User lulusan luar negeri, tanya dokumen apa yang diperlukan", "complexity": "reasoning"},
    {"id": "D17", "stage": "dokumen", "scenario": "User mau ganti foto setelah upload", "complexity": "edge_case"},
    {"id": "D18", "stage": "dokumen", "scenario": "User tanya apakah ijazah harus dilegalisir", "complexity": "reasoning"},
    {"id": "D19", "stage": "dokumen", "scenario": "User tanya dokumen tambahan untuk S2", "complexity": "direct"},
    {"id": "D20", "stage": "dokumen", "scenario": "User upload dokumen orang lain, minta koreksi", "complexity": "edge_case"},
    
    # ==========================================================================
    # TAHAP 5: PLACEMENT TEST (15 skenario)
    # ==========================================================================
    {"id": "T01", "stage": "placement_test", "scenario": "User tanya kapan bisa ujian placement test", "complexity": "direct"},
    {"id": "T02", "stage": "placement_test", "scenario": "User tanya durasi ujian placement test", "complexity": "direct"},
    {"id": "T03", "stage": "placement_test", "scenario": "User internet mati saat ujian", "complexity": "edge_case"},
    {"id": "T04", "stage": "placement_test", "scenario": "Menu ujian tidak muncul padahal dokumen sudah diupload", "complexity": "edge_case"},
    {"id": "T05", "stage": "placement_test", "scenario": "User tanya materi apa yang diujikan", "complexity": "reasoning"},
    {"id": "T06", "stage": "placement_test", "scenario": "User tanya apakah ujian bisa diulang", "complexity": "reasoning"},
    {"id": "T07", "stage": "placement_test", "scenario": "User laptop mati saat ujian berlangsung", "complexity": "edge_case"},
    {"id": "T08", "stage": "placement_test", "scenario": "User tanya tips mengerjakan placement test", "complexity": "reasoning"},
    {"id": "T09", "stage": "placement_test", "scenario": "User tanya bentuk soal placement test", "complexity": "direct"},
    {"id": "T10", "stage": "placement_test", "scenario": "User tidak sengaja submit sebelum selesai", "complexity": "edge_case"},
    {"id": "T11", "stage": "placement_test", "scenario": "User tanya passing grade placement test", "complexity": "reasoning"},
    {"id": "T12", "stage": "placement_test", "scenario": "User tanya jadwal ujian placement test", "complexity": "direct"},
    {"id": "T13", "stage": "placement_test", "scenario": "User mau reschedule jadwal ujian", "complexity": "edge_case"},
    {"id": "T14", "stage": "placement_test", "scenario": "User tanya apakah ujian wajib dikerjakan", "complexity": "direct"},
    {"id": "T15", "stage": "placement_test", "scenario": "User tanya hasil placement test keluar kapan", "complexity": "direct"},
    
    # ==========================================================================
    # TAHAP 6: PENGUMUMAN (15 skenario)
    # ==========================================================================
    {"id": "A01", "stage": "pengumuman", "scenario": "User tanya cara melihat hasil pengumuman", "complexity": "direct"},
    {"id": "A02", "stage": "pengumuman", "scenario": "User dapat status WAITING", "complexity": "reasoning"},
    {"id": "A03", "stage": "pengumuman", "scenario": "User diterima tapi deadline tinggal 2 hari", "complexity": "edge_case"},
    {"id": "A04", "stage": "pengumuman", "scenario": "User tanya kapan pengumuman keluar", "complexity": "direct"},
    {"id": "A05", "stage": "pengumuman", "scenario": "User dapat status DITOLAK, tanya opsi selanjutnya", "complexity": "reasoning"},
    {"id": "A06", "stage": "pengumuman", "scenario": "User tidak menemukan namanya di pengumuman", "complexity": "edge_case"},
    {"id": "A07", "stage": "pengumuman", "scenario": "User tanya arti status DITERIMA", "complexity": "direct"},
    {"id": "A08", "stage": "pengumuman", "scenario": "User tanya langkah setelah diterima", "complexity": "reasoning"},
    {"id": "A09", "stage": "pengumuman", "scenario": "User dapat 2 status berbeda untuk 2 prodi", "complexity": "edge_case"},
    {"id": "A10", "stage": "pengumuman", "scenario": "User tanya apakah pengumuman dikirim via email", "complexity": "direct"},
    {"id": "A11", "stage": "pengumuman", "scenario": "User mau konfirmasi kelulusan tapi tombol tidak aktif", "complexity": "edge_case"},
    {"id": "A12", "stage": "pengumuman", "scenario": "User tanya beda pengumuman tiap gelombang", "complexity": "reasoning"},
    {"id": "A13", "stage": "pengumuman", "scenario": "User lupa login untuk cek pengumuman", "complexity": "edge_case"},
    {"id": "A14", "stage": "pengumuman", "scenario": "User tanya apakah bisa banding jika ditolak", "complexity": "reasoning"},
    {"id": "A15", "stage": "pengumuman", "scenario": "User tanya syarat minimum untuk diterima", "complexity": "reasoning"},
    
    # ==========================================================================
    # TAHAP 7: REGISTRASI / PEMBAYARAN SEMESTER (20 skenario)
    # ==========================================================================
    {"id": "R01", "stage": "registrasi", "scenario": "User tanya biaya angsuran 1 semester D3/S1", "complexity": "direct"},
    {"id": "R02", "stage": "registrasi", "scenario": "User tanya biaya angsuran 1 semester S2", "complexity": "direct"},
    {"id": "R03", "stage": "registrasi", "scenario": "User tanya cara pembayaran semester 1", "complexity": "reasoning"},
    {"id": "R04", "stage": "registrasi", "scenario": "User bayar semester tapi status tidak update", "complexity": "edge_case"},
    {"id": "R05", "stage": "registrasi", "scenario": "User tanya rincian biaya kuliah lengkap", "complexity": "direct"},
    {"id": "R06", "stage": "registrasi", "scenario": "User tanya deadline pembayaran semester 1", "complexity": "direct"},
    {"id": "R07", "stage": "registrasi", "scenario": "User mau cicil pembayaran semester", "complexity": "reasoning"},
    {"id": "R08", "stage": "registrasi", "scenario": "User telat bayar semester 1, tanya konsekuensi", "complexity": "edge_case"},
    {"id": "R09", "stage": "registrasi", "scenario": "User tanya apakah ada potongan untuk bayar lunas", "complexity": "reasoning"},
    {"id": "R10", "stage": "registrasi", "scenario": "User tanya jadwal cicilan biaya kuliah", "complexity": "direct"},
    {"id": "R11", "stage": "registrasi", "scenario": "User mau bayar lebih dari angsuran 1", "complexity": "reasoning"},
    {"id": "R12", "stage": "registrasi", "scenario": "User salah transfer nominal semester", "complexity": "edge_case"},
    {"id": "R13", "stage": "registrasi", "scenario": "User tanya komponen biaya pendidikan", "complexity": "direct"},
    {"id": "R14", "stage": "registrasi", "scenario": "User tanya apakah biaya sama untuk semua prodi", "complexity": "reasoning"},
    {"id": "R15", "stage": "registrasi", "scenario": "User mau mundur setelah bayar semester 1", "complexity": "edge_case"},
    {"id": "R16", "stage": "registrasi", "scenario": "User tanya biaya untuk mahasiswa transfer", "complexity": "reasoning"},
    {"id": "R17", "stage": "registrasi", "scenario": "User tanya cara dapat bukti pembayaran", "complexity": "direct"},
    {"id": "R18", "stage": "registrasi", "scenario": "User pembayaran terblokir karena limit harian", "complexity": "edge_case"},
    {"id": "R19", "stage": "registrasi", "scenario": "User tanya apakah bisa bayar via e-wallet", "complexity": "reasoning"},
    {"id": "R20", "stage": "registrasi", "scenario": "User tanya perbedaan biaya reguler vs ekstension", "complexity": "direct"},
    
    # ==========================================================================
    # TAHAP 8: NIM (15 skenario)
    # ==========================================================================
    {"id": "N01", "stage": "nim", "scenario": "User tanya cara reservasi NIM", "complexity": "direct"},
    {"id": "N02", "stage": "nim", "scenario": "User belum dapat NIM setelah 3 hari reservasi", "complexity": "edge_case"},
    {"id": "N03", "stage": "nim", "scenario": "User tanya lama proses verifikasi NIM", "complexity": "direct"},
    {"id": "N04", "stage": "nim", "scenario": "User tanya yang harus dilakukan setelah dapat NIM", "complexity": "reasoning"},
    {"id": "N05", "stage": "nim", "scenario": "Tombol reservasi NIM tidak aktif", "complexity": "edge_case"},
    {"id": "N06", "stage": "nim", "scenario": "User tanya format NIM UNSIQ", "complexity": "direct"},
    {"id": "N07", "stage": "nim", "scenario": "User tanya fungsi NIM untuk mahasiswa", "complexity": "reasoning"},
    {"id": "N08", "stage": "nim", "scenario": "User NIM salah ketik, minta koreksi", "complexity": "edge_case"},
    {"id": "N09", "stage": "nim", "scenario": "User tanya cara ambil KTM setelah dapat NIM", "complexity": "direct"},
    {"id": "N10", "stage": "nim", "scenario": "User tanya jadwal orientasi mahasiswa baru", "complexity": "reasoning"},
    {"id": "N11", "stage": "nim", "scenario": "User tanya cara daftar matakuliah setelah dapat NIM", "complexity": "reasoning"},
    {"id": "N12", "stage": "nim", "scenario": "User sudah dapat NIM tapi belum dapat email konfirmasi", "complexity": "edge_case"},
    {"id": "N13", "stage": "nim", "scenario": "User tanya apakah NIM bisa berubah jika pindah prodi", "complexity": "reasoning"},
    {"id": "N14", "stage": "nim", "scenario": "User tanya apa itu portal akademik", "complexity": "direct"},
    {"id": "N15", "stage": "nim", "scenario": "User tanya cara akses SIAKAD setelah dapat NIM", "complexity": "direct"},
    
    # ==========================================================================
    # GENERAL (15 skenario)
    # ==========================================================================
    {"id": "G01", "stage": "general", "scenario": "User tanya alur pendaftaran lengkap", "complexity": "reasoning"},
    {"id": "G02", "stage": "general", "scenario": "User mau pindah prodi sebelum ujian", "complexity": "reasoning"},
    {"id": "G03", "stage": "general", "scenario": "User mau pindah prodi setelah dapat NIM", "complexity": "edge_case"},
    {"id": "G04", "stage": "general", "scenario": "User tanya kontak PMB", "complexity": "direct"},
    {"id": "G05", "stage": "general", "scenario": "User tanya jam operasional PMB", "complexity": "direct"},
    {"id": "G06", "stage": "general", "scenario": "User tanya apakah semua proses bisa online", "complexity": "direct"},
    {"id": "G07", "stage": "general", "scenario": "User tanya alamat kampus UNSIQ", "complexity": "direct"},
    {"id": "G08", "stage": "general", "scenario": "User tanya berapa lama proses pendaftaran total", "complexity": "reasoning"},
    {"id": "G09", "stage": "general", "scenario": "User mau batalkan pendaftaran", "complexity": "edge_case"},
    {"id": "G10", "stage": "general", "scenario": "User tanya email resmi PMB", "complexity": "direct"},
    {"id": "G11", "stage": "general", "scenario": "User mau konsultasi langsung ke PMB", "complexity": "reasoning"},
    {"id": "G12", "stage": "general", "scenario": "User complaint layanan PMB", "complexity": "edge_case"},
    {"id": "G13", "stage": "general", "scenario": "User tanya social media resmi PMB", "complexity": "direct"},
    {"id": "G14", "stage": "general", "scenario": "User tanya apakah ada open house atau campus tour", "complexity": "reasoning"},
    {"id": "G15", "stage": "general", "scenario": "User tanya dokumen apa saja yang harus disiapkan dari awal", "complexity": "reasoning"},
    
    # ==========================================================================
    # JADWAL GELOMBANG (15 skenario)
    # ==========================================================================
    {"id": "JG01", "stage": "jadwal", "scenario": "User tanya periode pendaftaran Gelombang 1, 2, dan 3", "complexity": "direct"},
    {"id": "JG02", "stage": "jadwal", "scenario": "User tanya jadwal ujian tiap gelombang", "complexity": "direct"},
    {"id": "JG03", "stage": "jadwal", "scenario": "User tanya deadline pembayaran angsuran 1 tiap gelombang", "complexity": "direct"},
    {"id": "JG04", "stage": "jadwal", "scenario": "User tanya kapan kuliah dimulai jika daftar Gelombang 2", "complexity": "reasoning"},
    {"id": "JG05", "stage": "jadwal", "scenario": "User bingung pilih Gelombang 1 atau 2", "complexity": "reasoning"},
    {"id": "JG06", "stage": "jadwal", "scenario": "User tanya keuntungan daftar Gelombang 1", "complexity": "reasoning"},
    {"id": "JG07", "stage": "jadwal", "scenario": "User telat daftar Gelombang 1, tanya opsi", "complexity": "edge_case"},
    {"id": "JG08", "stage": "jadwal", "scenario": "User tanya apakah kuota tiap gelombang berbeda", "complexity": "reasoning"},
    {"id": "JG09", "stage": "jadwal", "scenario": "User tanya jadwal lengkap PMB tahun ini", "complexity": "direct"},
    {"id": "JG10", "stage": "jadwal", "scenario": "User mau pindah gelombang setelah daftar", "complexity": "edge_case"},
    {"id": "JG11", "stage": "jadwal", "scenario": "User tanya kapan terakhir bisa daftar tahun ini", "complexity": "direct"},
    {"id": "JG12", "stage": "jadwal", "scenario": "User belum lulus SMA, tanya gelombang mana yang cocok", "complexity": "reasoning"},
    {"id": "JG13", "stage": "jadwal", "scenario": "User tanya perbedaan jadwal D3/S1 dan S2", "complexity": "direct"},
    {"id": "JG14", "stage": "jadwal", "scenario": "User tanya apakah ada gelombang khusus untuk pindahan", "complexity": "reasoning"},
    {"id": "JG15", "stage": "jadwal", "scenario": "User mau daftar tahun depan, tanya jadwal", "complexity": "edge_case"},
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
❌ "Hai Kak! Wah keren banget mau daftar ya..."
✅ "Baik, berikut dokumen yang diperlukan: 1. Ijazah/SKL 2. Kartu Keluarga 3. Foto formal"
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
    print("ALUR PENDAFTARAN DATASET GENERATOR (FORMAL STYLE)")
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
    output_file = os.path.join(output_dir, "multiturn_alur_pendaftaran_100.json")
    
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
            context = CONTEXT_BY_STAGE.get(scenario["stage"], CONTEXT_BY_STAGE["general"])
            
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
                    "instruction": f"Multi-turn conversation about UNSIQ registration - {scenario['stage']}",
                    "input": "",
                    "output": json.dumps(conversation, ensure_ascii=False),
                    "text": "",
                    "category": "alur_pendaftaran",
                    "stage": scenario["stage"],
                    "scenario": scenario["scenario"],
                    "complexity": scenario["complexity"],
                    "source": "synthetic_alur_v2_formal"
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
