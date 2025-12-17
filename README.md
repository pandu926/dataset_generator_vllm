# 🚀 UNSIQ Synthetic Dataset Pipeline

> **START HERE (PENTING):**
> Sebelum menjalankan kode, **WAJIB** membaca dokumen di folder `docs/` untuk memahami strategi dan tujuan project ini.

## 📖 Must-Read Documentation

Jika Anda baru di project ini, buka file berikut secara berurutan:

1.  **`docs/PROJECT_BLUEPRINT.md`** (👑 Master Plan): Menjelaskan visi, arsitektur, dan standar kualitas dataset yang kita bangun.
2.  **`docs/synthetic_data_strategy.md`**: Detail teknis tentang strategi "Anti-Forgetting", Persona, dan Complexity Tiers.

---

## 🏗️ Project Structure

File dan folder dalam project ini telah diatur dengan standar _Python Project_ sebagai berikut:

```text
pipeline-dataset-generator/
│
├── 📂 docs/               # 🧠 OTAK PROJECT (Dokumentasi & Strategi)
│   ├── PROJECT_BLUEPRINT.md      # Master Plan
│   └── synthetic_data_strategy.md # Strategi Detail
│
├── 📂 data/               # 🗄️ GUDANG DATA
│   ├── seeds/             # Input: File JSON lama (dataset_biaya.json, dll)
│   ├── rag_source/        # Input: Dokumen PDF/MD sumber kebenaran (RAG)
│   ├── chunks/            # Process: Hasil potongan RAG siap pakai
│   └── output/            # Result: Dataset final hasil generate
│
├── 📂 src/                # ⚙️ MESIN (Source Code)
│   ├── llm_multiturn_generator.py # Logic utama generator (Persona, CoT)
│   ├── vllm_engine.py             # Driver untuk koneksi ke Model LLM
│   ├── phase1_preparation.py      # Script pembuat Chunks RAG
│   └── e5_embedding.py            # Script embedding search
│
├── 📂 config/             # 🛠️ PENGATURAN
│   └── config.yaml        # Konfigurasi path, model, dan prompt
│
├── main.py                # ▶️ TOMBOL START (Script Eksekusi Utama)
└── requirements.txt       # Daftar library python
```

---

## 🎯 Project Goal

Tujuan project ini adalah men-transformasi data tanya-jawab pendek (Single-turn) menjadi percakapan panjang yang cerdas (Multi-turn) untuk melatih model AI (Fine-tuning Gemma-3-1B).

**Output yang diharapkan:**
1.000 Percakapan JSONL yang memiliki:

- **Reasoning**: Bot berpikir sebelum menjawab.
- **Persona**: Bot memiliki karakter (bukan robot kaku).
- **Clarification**: Bot bertanya balik jika user tidak jelas.

---

## 🚀 How to Run

1.  Pastikan semua library terinstall.
2.  Siapkan seed data di `data/seeds/`.
3.  Jalankan perintah:
    ```bash
    python main.py
    ```

---

_Dibuat oleh Tim Agentic Coding untuk Project UNSIQ Fine-tuning._
