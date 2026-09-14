# AI Expert Finder: NLP & Vector Retrieval Matching Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-LangChain-orange.svg)](https://www.langchain.com/)
[![VectorDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-green.svg)](https://www.trychroma.com/)
[![Embeddings](https://img.shields.io/badge/Embeddings-SentenceTransformers-yellow.svg)](https://sbert.net/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> A privacy-conscious semantic retrieval and LLM-as-a-Judge re-ranking engine designed to eliminate keyword mismatch and surface overlooked technical talent.

---

## 📌 Executive Summary
Traditional Applicant Tracking Systems (ATS) rely heavily on exact keyword matching, creating a critical "semantic gap" where qualified candidates with non-standard phrasing are filtered out. **AI Expert Finder** solves this using a two-stage hybrid pipeline:
1. **Dense Vector Retrieval**: High-dimensional semantic embeddings (`sentence-transformers/all-MiniLM-L6-v2`) indexed in ChromaDB.
2. **LLM-as-a-Judge Evaluation**: Quantized local LLMs (via Ollama / Llama2) to extract skills and evaluate candidate fit with transparent justifications.

---

## 📊 Benchmark Results

Evaluated across **13,389 resumes** spanning **43 job categories**:

| Category | Query Specificity | MAP@5 | Recall@5 | NDCG@5 | Relevant Found (/5) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **React Developer** | Moderate | **1.000** | **1.000** | **1.000** | 5/5 |
| **Consultant** | Moderate | **1.000** | **1.000** | **1.000** | 5/5 |
| **Java Developer** | Moderate | **1.000** | **1.000** | **1.000** | 5/5 |
| **Human Resources** | Moderate | **1.000** | **1.000** | **1.000** | 5/5 |
| **System Average** | Across All Queries | **0.736** | **0.720** | **0.785** | - |

**Key Finding**: Moderate query specificity consistently outperformed vague and over-constrained queries, achieving perfect 1.000 MAP@5 across key technical categories.

---

## 📂 Repository Structure
```text
ai-expert-finder/
├── src/
│   ├── preprocessing.py    # PII masking (emails, phones, URLs)
│   ├── vector_store.py     # ChromaDB dense retrieval
│   ├── llm_judge.py        # LLM skill extraction & re-ranking
│   └── evaluation.py       # IR metrics (MAP@k, NDCG@k)
├── data/                   # Schema documentation & sample data
├── benchmarks/             # Evaluation logs & metrics CSV
├── demo.py                 # Interactive CLI pipeline demo
├── requirements.txt        # Python dependencies
└── LICENSE                 # MIT License
