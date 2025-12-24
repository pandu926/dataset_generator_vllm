
## 4.4. Evaluasi Model: BERTScore Analysis

Evaluasi pertama dilakukan menggunakan **BERTScore**, metrik evaluasi yang menghitung kesamaan semantik antara respons model dan jawaban referensi menggunakan embeddings BERT pre-trained (DeBERTa-XLarge-MNLI). BERTScore lebih robust dibanding metrik berbasis n-gram (seperti BLEU) karena dapat menangkap kesamaan semantik meskipun kata-katanya berbeda.

### 4.4.1. Performa Agregat BERTScore

Tabel 4.4 menunjukkan hasil BERTScore agregat untuk base model (Gemma 3-1B tanpa fine-tuning) dan fine-tuned model, dievaluasi pada **175 test conversations**:

| Metrik        | Base Model | Fine-tuned | Peningkatan |
| ------------- | ---------- | ---------- | ----------- |
| **Precision** | 0.5469     | 0.6804     | +24.4%      |
| **Recall**    | 0.6406     | 0.6697     | +4.5%       |
| **F1-Score**  | 0.5891     | 0.6731     | +14.2%      |

**Interpretasi:**

- Base model mencapai F1-Score **0.5891** (58.91%), menunjukkan model general-purpose sudah memiliki kemampuan linguistik dasar namun belum optimal untuk domain PMB UNSIQ.
- Fine-tuned model mencapai F1-Score **0.6731** (67.31%), peningkatan sebesar **14.2%** atau **+0.084 poin absolute**.
- Peningkatan Precision (+24.4%) jauh lebih besar dari Recall (+4.5%), menandakan model fine-tuned lebih **presisi** dalam memilih kata yang tepat, meski tidak selalu mencakup semua informasi yang diharapkan.

---

## 4.5. Evaluasi Model: LLM-as-Judge Analysis

Evaluasi kedua menggunakan **LLM-as-Judge dengan RAG**, pendekatan evaluasi kualitatif dimana model besar (Gemma 3-12B) bertindak sebagai "judge" dengan akses ke konteks referensi untuk menilai kualitas respons berdasarkan 6 dimensi.

### 4.5.1. Performa Agregat LLM-as-Judge

Tabel 4.6 menunjukkan hasil LLM-as-Judge agregat untuk base dan fine-tuned model, dievaluasi pada **175 test conversations**:

| Dimensi Penilaian        | Base Model  | Fine-tuned  | Peningkatan        |
| ------------------------ | ----------- | ----------- | ------------------ |
| **Helpfulness**          | 2.38 / 5.00 | 2.99 / 5.00 | +0.62 (+25.9%)     |
| **Relevance**            | 3.68 / 5.00 | 4.28 / 5.00 | +0.60 (+16.3%)     |
| **Factual Accuracy**     | 1.60 / 5.00 | 2.16 / 5.00 | +0.55 (+34.4%)     |
| **Hallucination Check**  | 1.66 / 5.00 | 2.14 / 5.00 | +0.48 (+28.6%)     |
| **Coherence**            | 3.63 / 5.00 | 4.29 / 5.00 | +0.66 (+18.2%)     |
| **Fluency**              | 4.53 / 5.00 | 4.83 / 5.00 | +0.30 (+6.6%)      |
| **Overall Score**        | **2.78**    | **3.28**    | **+0.49** (+17.7%) |

**Interpretasi Agregat:**

- Base model mencapai overall score **2.78/5.00**, menunjukkan model tanpa fine-tuning sering memberikan jawaban yang tidak akurat dan mengandung halusinasi.
- Fine-tuned model mencapai overall score **3.28/5.00**, peningkatan sebesar **17.7%** atau **+0.49 poin absolute**.
- **Temuan kritis**: Fluency sudah tinggi di base model (4.53) dan bahkan lebih tinggi di fine-tuned (4.83), namun **Factual Accuracy tetap rendah** (2.16/5.00).

### 4.5.2. Analisis Heterogenitas: Pola Peningkatan yang Tidak Merata

**Temuan kritis**: Peningkatan tidak merata di berbagai dimensi:

| Dimensi | Improvement | Kategori |
|---------|-------------|----------|
| Factual Accuracy | +34.4% | Faktual |
| Hallucination Check | +28.6% | Faktual |
| Helpfulness | +25.9% | Linguistik |
| Coherence | +18.2% | Linguistik |
| Relevance | +16.3% | Linguistik |
| Fluency | +6.6% | Linguistik |

**Interpretasi:**

> "Fine-tuning memberikan peningkatan terbesar pada **Factual Accuracy** (+34.4%) dan **Hallucination Check** (+28.6%), menunjukkan model berhasil belajar mengurangi halusinasi. Namun, skor absolutnya tetap **rendah** (2.16/5.00), menandakan fine-tuning saja tidak cukup untuk menjamin akurasi faktual yang tinggi."

---

## 4.6. Perbandingan Base vs Fine-tuned: Analisis Mendalam

### 4.6.1. Ringkasan Perbandingan Agregat

| Aspek Evaluasi          | Base  | Fine-tuned | Δ Absolute | Δ Pct   |
| ----------------------- | ----- | ---------- | ---------- | ------- |
| **BERTScore F1**        | 0.589 | 0.673      | +0.084     | +14.2%  |
| **Helpfulness**         | 2.38  | 2.99       | +0.62      | +25.9%  |
| **Relevance**           | 3.68  | 4.28       | +0.60      | +16.3%  |
| **Factual Accuracy**    | 1.60  | 2.16       | +0.55      | +34.4%  |
| **Hallucination Check** | 1.66  | 2.14       | +0.48      | +28.6%  |
| **Coherence**           | 3.63  | 4.29       | +0.66      | +18.2%  |
| **Fluency**             | 4.53  | 4.83       | +0.30      | +6.6%   |
| **Overall LLM Score**   | 2.78  | 3.28       | +0.49      | +17.7%  |

### 4.6.2. Contoh Evaluasi Per Kategori

#### Kategori: Out-of-Topic (OOT)
**Pertanyaan**: "Eh, tau gak sih harga cabai sekarang berapa di pasar?"

| Model | Response | Accuracy | Hallucination |
|-------|----------|----------|---------------|
| **Base** | Memberikan harga cabai Jakarta/Bandung/Surabaya (halusinasi) | 1/5 | 1/5 |
| **Fine-tuned** | "Sayangnya, informasi harga cabai tidak tersedia..." | 5/5 | 5/5 |

**Insight**: Model fine-tuned belajar untuk **mengakui keterbatasan** daripada mengarang informasi.

---

#### Kategori: Syarat & Ketentuan (SNK)
**Pertanyaan**: "Biaya pendaftaran untuk gelombang pertama tahun 2025?"

| Model | Response | Accuracy |
|-------|----------|----------|
| **Base** | Menjelaskan sistem pendaftaran dengan halusinasi | 1/5 |
| **Fine-tuned** | "Biaya pendaftarannya adalah Rp 250.000" | 1/5 |
| **Ground Truth** | Rp 0 (Gratis) | - |

**Insight**: Model fine-tuned tetap **salah dalam angka spesifik**, menunjukkan keterbatasan parametric memory.

---

#### Kategori: Beasiswa
**Pertanyaan**: "Apakah ada beasiswa untuk Ustadz/Ustadzah TPQ & Madin?"

| Model | Response | Score |
|-------|----------|-------|
| **Base** | Menjelaskan beasiswa dari pondok pesantren (halusinasi) | 3.33/5 |
| **Fine-tuned** | "Saya sarankan menghubungi bagian kemahasiswaan..." | 3.00/5 |

**Insight**: Kedua model tidak memberikan jawaban spesifik dari konteks UNSIQ.

---

### 4.6.3. The Confidence Trap: Temuan Kritis

**Fenomena Utama:**

Model fine-tuned mencapai **Fluency 4.83/5.00** dan **Coherence 4.29/5.00** (terdengar sangat profesional), namun **Factual Accuracy hanya 2.16/5.00** (sering salah).

**Gap Clarity vs Accuracy**: ~2.67 poin

| Tipe Pertanyaan | Fluency | Accuracy | Gap | Risk |
|-----------------|---------|----------|-----|------|
| Out-of-Topic | 4.8 | 3.3+ | 1.5 | � LOW (sering tolak dengan benar) |
| Fasilitas | 4.9 | 3.8 | 1.1 | � LOW |
| Biaya/SNK | 4.8 | 1-2 | 2.8+ | 🔴 CRITICAL |
| Beasiswa | 4.9 | 2-3 | 1.9 | 🟠 HIGH |

**Implikasi:**

> "Model berbicara dengan sangat percaya diri dan profesional, tetapi **informasi faktual sering salah**. Untuk kategori numerik (biaya, jadwal), diperlukan **RAG** untuk verifikasi."

---

### 4.6.4. Kesimpulan Analisis

**Positif:**
1. ✅ Model fine-tuned lebih **relevan** (+16.3%) dan **koheren** (+18.2%)
2. ✅ Model belajar **mengakui keterbatasan** untuk pertanyaan di luar topik
3. ✅ **Halusinasi berkurang** (+28.6% improvement)

**Negatif:**
1. ❌ **Factual accuracy tetap rendah** (2.16/5.00)
2. ❌ **Angka spesifik** (biaya, jadwal) sering salah
3. ❌ **Confidence trap**: model terdengar meyakinkan meski salah

**Rekomendasi:**
1. Gunakan **RAG** untuk pertanyaan faktual (biaya, jadwal, persyaratan)
2. Implementasi **confidence threshold** untuk eskalasi ke human
3. Model cocok untuk **intent understanding** dan **routing**, bukan sebagai sumber kebenaran tunggal

---

## 4.7. Hasil Evaluasi Per Kategori (LLM-as-Judge)

Berikut adalah breakdown skor LLM-as-Judge untuk setiap kategori:

### 4.7.1. BEASISWA (n=29)

| Dimensi | Base | Fine-tuned | Δ |
|---------|------|------------|---|
| helpfulness | 2.83 | 3.34 | +0.52 |
| relevance | 3.93 | 4.69 | +0.76 |
| factual_accuracy | 1.79 | **2.93** | **+1.14** |
| hallucination_check | 1.93 | 2.86 | +0.93 |
| coherence | 3.79 | 4.41 | +0.62 |
| fluency | 4.72 | 4.90 | +0.17 |
| **total** | **3.11** | **3.77** | **+0.66** |

**Insight**: Kategori dengan peningkatan **factual_accuracy tertinggi** (+1.14). Model berhasil belajar informasi beasiswa.

---

### 4.7.2. BIAYA (n=32)

| Dimensi | Base | Fine-tuned | Δ |
|---------|------|------------|---|
| helpfulness | 1.91 | 2.94 | +1.04 |
| relevance | 3.62 | 4.46 | +0.83 |
| factual_accuracy | 1.28 | **1.46** | +0.18 |
| hallucination_check | 1.31 | 1.49 | +0.17 |
| coherence | 3.59 | 4.49 | +0.89 |
| fluency | 4.47 | 4.97 | +0.50 |
| **total** | **2.49** | **2.91** | **+0.42** |

**Insight**: Kategori dengan **accuracy terendah** (1.46/5). Model sering salah dalam menyebut angka biaya kuliah. **⚠️ CRITICAL**

---

### 4.7.3. FASILITAS (n=25)

| Dimensi | Base | Fine-tuned | Δ |
|---------|------|------------|---|
| helpfulness | 2.60 | 3.16 | +0.56 |
| relevance | 4.00 | 4.44 | +0.44 |
| factual_accuracy | 1.80 | 2.56 | +0.76 |
| hallucination_check | 1.76 | 2.52 | +0.76 |
| coherence | 3.80 | 4.40 | +0.60 |
| fluency | 4.52 | 4.96 | +0.44 |
| **total** | **2.97** | **3.56** | **+0.59** |

**Insight**: Peningkatan konsisten di semua dimensi. Informasi prosedural lebih mudah dipelajari.

---

### 4.7.4. PRODI (n=27)

| Dimensi | Base | Fine-tuned | Δ |
|---------|------|------------|---|
| helpfulness | 2.70 | 3.31 | +0.61 |
| relevance | 4.22 | 4.55 | +0.33 |
| factual_accuracy | 1.59 | 2.14 | +0.55 |
| hallucination_check | 1.70 | 2.10 | +0.40 |
| coherence | 3.74 | 4.41 | +0.67 |
| fluency | 4.63 | 4.86 | +0.23 |
| **total** | **2.90** | **3.44** | **+0.54** |

**Insight**: Model belajar baik tentang program studi, meski kadang salah menyebut nama prodi atau akreditasi.

---

### 4.7.5. SNK - Syarat & Ketentuan (n=32)

| Dimensi | Base | Fine-tuned | Δ |
|---------|------|------------|---|
| helpfulness | 2.53 | 3.16 | +0.62 |
| relevance | 3.78 | 4.53 | +0.75 |
| factual_accuracy | 1.81 | 2.22 | +0.41 |
| hallucination_check | 1.88 | 2.19 | +0.31 |
| coherence | 3.56 | 4.34 | +0.78 |
| fluency | 4.53 | 4.88 | +0.34 |
| **total** | **2.93** | **3.44** | **+0.51** |

**Insight**: Peningkatan moderat. Model sering salah dalam detail spesifik (tanggal, biaya per gelombang).

---

### 4.7.6. ALUR_PENDAFTARAN (n=4)

| Dimensi | Base | Fine-tuned | Δ |
|---------|------|------------|---|
| helpfulness | 2.00 | 2.50 | +0.50 |
| relevance | 3.50 | 4.50 | +1.00 |
| factual_accuracy | 1.25 | 1.75 | +0.50 |
| hallucination_check | 1.50 | 1.75 | +0.25 |
| coherence | 3.75 | 4.00 | +0.25 |
| fluency | 4.75 | 5.00 | +0.25 |
| **total** | **2.67** | **3.12** | **+0.46** |

**Insight**: Sample size kecil. Relevance meningkat signifikan (+1.00).

---

### 4.7.7. OUT_OF_TOPIC (n=12)

| Dimensi | Base | Fine-tuned | Δ |
|---------|------|------------|---|
| helpfulness | 1.17 | 1.42 | +0.25 |
| relevance | 1.42 | 1.67 | +0.25 |
| factual_accuracy | 1.08 | **1.75** | **+0.67** |
| hallucination_check | 1.17 | **1.92** | **+0.75** |
| coherence | 3.00 | 3.50 | +0.50 |
| fluency | 4.08 | 4.42 | +0.33 |
| **total** | **1.82** | **2.25** | **+0.43** |

**Insight**: Model fine-tuned lebih baik dalam **menolak pertanyaan di luar topik** dan mengurangi halusinasi (+0.75).

---

### 4.7.8. INFORMASI_UMUM (n=7)

| Dimensi | Base | Fine-tuned | Δ |
|---------|------|------------|---|
| helpfulness | 2.14 | 2.00 | **-0.14** |
| relevance | 3.00 | 3.14 | +0.14 |
| factual_accuracy | 1.71 | 1.57 | **-0.14** |
| hallucination_check | 1.57 | 1.57 | +0.00 |
| coherence | 3.43 | 3.14 | **-0.29** |
| fluency | 4.29 | 3.71 | **-0.57** |
| **total** | **2.62** | **2.38** | **-0.24** |

**Insight**: ⚠️ **Kategori dengan penurunan skor**. Kemungkinan karena sample size kecil (n=7) atau variasi pertanyaan.

---

### 4.7.9. PROFIL_UNSIQ (n=1)

| Dimensi | Base | Fine-tuned | Δ |
|---------|------|------------|---|
| helpfulness | 3.00 | 4.00 | +1.00 |
| relevance | 5.00 | 5.00 | +0.00 |
| factual_accuracy | 2.00 | 3.00 | +1.00 |
| coherence | 4.00 | 5.00 | +1.00 |
| fluency | 5.00 | 5.00 | +0.00 |
| **total** | **3.50** | **4.00** | **+0.50** |

**Insight**: Hanya 1 sample, tidak representatif.

---

### 4.7.10. Ringkasan Per Kategori

| Kategori | n | Base Total | FT Total | Δ | Status |
|----------|---|------------|----------|---|--------|
| Profil UNSIQ | 1 | 3.50 | 4.00 | +0.50 | ⚪ n=1 |
| Beasiswa | 29 | 3.11 | **3.77** | **+0.66** | 🟢 BEST |
| Fasilitas | 25 | 2.97 | 3.56 | +0.59 | 🟢 GOOD |
| Prodi | 27 | 2.90 | 3.44 | +0.54 | 🟢 GOOD |
| SNK | 32 | 2.93 | 3.44 | +0.51 | 🟡 OK |
| Alur Pendaftaran | 4 | 2.67 | 3.12 | +0.46 | 🟡 OK |
| Out-of-Topic | 12 | 1.82 | 2.25 | +0.43 | 🟡 OK |
| Biaya | 32 | 2.49 | **2.91** | +0.42 | 🔴 LOW ACC |
| Informasi Umum | 7 | 2.62 | 2.38 | **-0.24** | 🔴 REGRESS |

---