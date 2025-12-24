
## 4.4. Evaluasi Model: BERTScore Analysis

Evaluasi pertama dilakukan menggunakan **BERTScore**, metrik evaluasi yang menghitung kesamaan semantik antara respons model dan jawaban referensi menggunakan embeddings BERT pre-trained (DeBERTa-XLarge-MNLI). BERTScore lebih robust dibanding metrik berbasis n-gram (seperti BLEU) karena dapat menangkap kesamaan semantik meskipun kata-katanya berbeda.

### 4.4.1. Konfigurasi Training

| Parameter | Nilai |
|-----------|-------|
| **Base Model** | google/gemma-3-1b-it |
| **LoRA Rank** | 32 |
| **LoRA Alpha** | 64 |
| **Learning Rate** | 2e-4 |
| **Epochs** | 3 |
| **Batch Size** | 16 x 4 = 64 (effective) |
| **Max Seq Length** | 2048 |

### 4.4.2. Performa Agregat BERTScore

Tabel 4.4 menunjukkan hasil BERTScore agregat untuk base model (Gemma 3-1B tanpa fine-tuning) dan fine-tuned model, dievaluasi pada **207 test conversations**:

| Metrik        | Base Model | Fine-tuned | Peningkatan |
| ------------- | ---------- | ---------- | ----------- |
| **Precision** | 0.5487     | 0.6887     | +25.5%      |
| **Recall**    | 0.6460     | 0.6761     | +4.7%       |
| **F1-Score**  | 0.5924     | 0.6808     | +14.9%      |

**Interpretasi:**

- Base model mencapai F1-Score **0.5924** (59.24%), menunjukkan model general-purpose sudah memiliki kemampuan linguistik dasar namun belum optimal untuk domain PMB UNSIQ.
- Fine-tuned model mencapai F1-Score **0.6808** (68.08%), peningkatan sebesar **14.9%** atau **+0.088 poin absolute**.
- Peningkatan Precision (+25.5%) jauh lebih besar dari Recall (+4.7%), menandakan model fine-tuned lebih **presisi** dalam memilih kata yang tepat, meski tidak selalu mencakup semua informasi yang diharapkan.

---

## 4.5. Evaluasi Model: LLM-as-Judge Analysis

Evaluasi kedua menggunakan **LLM-as-Judge dengan RAG**, pendekatan evaluasi kualitatif dimana model besar (Gemma 3-12B) bertindak sebagai "judge" dengan akses ke konteks referensi untuk menilai kualitas respons berdasarkan 6 dimensi.

### 4.5.1. Performa Agregat LLM-as-Judge

Tabel 4.6 menunjukkan hasil LLM-as-Judge agregat untuk base dan fine-tuned model, dievaluasi pada **207 test conversations**:

| Dimensi Penilaian        | Base Model  | Fine-tuned  | Peningkatan        |
| ------------------------ | ----------- | ----------- | ------------------ |
| **Helpfulness**          | 2.27 / 5.00 | 3.00 / 5.00 | +0.73 (+32.4%)     |
| **Relevance**            | 3.62 / 5.00 | 4.20 / 5.00 | +0.58 (+16.0%)     |
| **Factual Accuracy**     | 1.50 / 5.00 | 2.26 / 5.00 | +0.76 (+50.7%)     |
| **Hallucination Check**  | 1.56 / 5.00 | 2.23 / 5.00 | +0.67 (+42.9%)     |
| **Coherence**            | 3.60 / 5.00 | 4.32 / 5.00 | +0.71 (+19.8%)     |
| **Fluency**              | 4.50 / 5.00 | 4.81 / 5.00 | +0.31 (+7.0%)      |
| **Overall Score**        | **2.71**    | **3.33**    | **+0.61** (+22.6%) |

**Interpretasi Agregat:**

- Base model mencapai overall score **2.71/5.00**, menunjukkan model tanpa fine-tuning sering memberikan jawaban yang tidak akurat dan mengandung halusinasi.
- Fine-tuned model mencapai overall score **3.33/5.00**, peningkatan sebesar **22.6%** atau **+0.61 poin absolute**.
- **Temuan positif**: Factual Accuracy meningkat **50.7%** dan Hallucination Check meningkat **42.9%** - jauh lebih baik dari training sebelumnya!

### 4.5.2. Perbandingan dengan Training Sebelumnya

| Training Config | BERTScore F1 | Overall LLM | Fact Acc |
|-----------------|--------------|-------------|----------|
| LoRA r=16, α=32, ep=4 | 0.673 | 3.28 | 2.16 |
| **LoRA r=32, α=64, ep=3** | **0.681** | **3.33** | **2.26** |
| Δ | +0.008 | +0.05 | +0.10 |

**Insight**: Training dengan LoRA rank yang lebih besar (32 vs 16) memberikan peningkatan pada **Factual Accuracy** (+0.10), meskipun dengan epoch lebih sedikit (3 vs 4).

---

## 4.6. Hasil Evaluasi Per Kategori (LLM-as-Judge)

Berikut adalah breakdown skor LLM-as-Judge untuk setiap kategori:

### 4.6.1. Ringkasan Per Kategori

| Kategori | n | Base | Fine-tuned | Δ | Status |
|----------|---|------|------------|-----|--------|
| informasi_umum | 15 | 2.10 | **3.06** | **+0.97** | 🟢 BEST |
| prodi | 29 | 2.98 | 3.82 | +0.84 | 🟢 |
| fasilitas | 25 | 2.97 | 3.72 | +0.75 | 🟢 |
| beasiswa | 27 | 3.06 | 3.73 | +0.67 | 🟢 |
| snk | 32 | 2.81 | 3.35 | +0.54 | 🟢 |
| profil_unsiq | 15 | 3.02 | 3.55 | +0.53 | 🟢 |
| alur_pendaftaran | 10 | 2.45 | 2.92 | +0.47 | 🟡 |
| biaya | 31 | 2.45 | 2.88 | +0.43 | 🟡 |
| out_of_topic | 15 | 1.94 | 2.26 | +0.32 | 🟡 |

### 4.6.2. Analisis Per Kategori

#### 🟢 INFORMASI_UMUM (n=15) - Peningkatan Terbaik: +0.97

Kategori dengan peningkatan tertinggi. Model berhasil belajar informasi umum tentang UNSIQ seperti jadwal, kontak, dan prosedur.

---

#### 🟢 PRODI (n=29) - Peningkatan: +0.84

Model belajar baik tentang program studi, fakultas, dan akreditasi. Relevance tinggi (4.55).

---

#### 🟢 FASILITAS (n=25) - Peningkatan: +0.75

Informasi prosedural tentang fasilitas kampus berhasil dipelajari dengan baik. Coherence tinggi (4.40).

---

#### 🟢 BEASISWA (n=27) - Peningkatan: +0.67

Kategori dengan skor absolute tertinggi (3.73). Model berhasil belajar jenis-jenis beasiswa yang tersedia.

---

#### 🟡 BIAYA (n=31) - Peningkatan: +0.43

Kategori dengan **accuracy terendah**. Model masih sering salah dalam menyebut angka biaya kuliah. **⚠️ Perlu RAG untuk verifikasi angka.**

---

#### 🟡 OUT_OF_TOPIC (n=15) - Peningkatan: +0.32

Model fine-tuned lebih baik dalam menolak pertanyaan di luar topik, tapi kadang masih tetap mencoba menjawab.

---

## 4.7. The Confidence Trap: Temuan Kritis

**Fenomena Utama:**

Model fine-tuned mencapai **Fluency 4.81/5.00** dan **Coherence 4.32/5.00** (terdengar sangat profesional), namun **Factual Accuracy hanya 2.26/5.00** (sering salah).

| Tipe Pertanyaan | Fluency | Accuracy | Gap | Risk |
|-----------------|---------|----------|-----|------|
| Out-of-Topic | 4.4 | 1.5-2 | ~2.5 | 🔴 CRITICAL |
| Biaya/SNK | 4.8 | 2.0-2.5 | ~2.5 | 🔴 CRITICAL |
| Beasiswa | 4.9 | 2.5-3 | ~2 | 🟠 HIGH |
| Fasilitas/Prodi | 4.9 | 3-3.5 | ~1.5 | 🟡 MEDIUM |

---

## 4.8. Kesimpulan dan Rekomendasi

### 4.8.1. Hasil Positif

1. ✅ **Overall score meningkat 22.6%** (2.71 → 3.33)
2. ✅ **Factual Accuracy meningkat 50.7%** - improvement signifikan
3. ✅ **Hallucination berkurang 42.9%**
4. ✅ **Semua kategori menunjukkan peningkatan** (tidak ada regresi)
5. ✅ **Informasi Umum meningkat dramatis** (+0.97)

### 4.8.2. Keterbatasan

1. ❌ **Factual accuracy tetap rendah** (2.26/5.00) untuk standar production
2. ❌ **Angka spesifik** (biaya, jadwal) sering salah
3. ❌ **Confidence trap**: model terdengar meyakinkan meski salah

### 4.8.3. Rekomendasi Deployment

| Skenario | Rekomendasi |
|----------|-------------|
| **Intent routing** | ✅ Cocok - relevance tinggi (4.20) |
| **Jawaban prosedural** | ✅ Cocok dengan disclaimer |
| **Jawaban numerik (biaya)** | ❌ Gunakan RAG wajib |
| **Production chatbot** | ⚠️ Butuh RAG + human escalation |

### 4.8.4. Konfigurasi Final

```
Model: google/gemma-3-1b-it + QLoRA
LoRA: rank=32, alpha=64
Training: 3 epochs, lr=2e-4, batch=64
Dataset: 1,438 train, 207 test
BERTScore F1: 0.681 (+14.9%)
LLM-Judge: 3.33/5.00 (+22.6%)
```

---