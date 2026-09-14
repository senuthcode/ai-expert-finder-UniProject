#!/usr/bin/env python3
"""
AI Expert Finder - Interactive Pipeline Demo.
Demonstrates preprocessing, semantic matching, and re-ranking on sample candidate resumes.
"""

import argparse
import sys
from src.preprocessing import ResumePreprocessor
from src.llm_judge import LLMReRanker

SAMPLE_CANDIDATES = [
    {
        "id": "cand_01",
        "category": "React Developer",
        "raw_resume": """
John Doe - Senior Frontend Engineer (john.doe@example.com, +1 555-0199)
Experience:
- 4 years building scalable web applications with React, TypeScript, Next.js, and Redux Toolkit.
- Architected REST and GraphQL API integrations, reducing client-side data fetch times by 35%.
- Implemented automated testing with Jest and Cypress, maintaining 85%+ code coverage.
- Optimized web vitals across core dashboards, cutting bundle size by 28%.
Skills: React, TypeScript, JavaScript, Next.js, Redux, TailwindCSS, Jest, Webpack.
"""
    },
    {
        "id": "cand_02",
        "category": "Data Science",
        "raw_resume": """
Jane Smith - Machine Learning Researcher (jane.smith@email.com, 0412345678)
Experience:
- Developed deep learning pipelines for NLP and semantic retrieval using PyTorch and Transformers.
- Fine-tuned BERT and Sentence-Transformer models on domain-specific corpora.
- Scaled vector search infrastructure using ChromaDB and FAISS for 500k+ documents.
Skills: Python, PyTorch, Hugging Face, Transformers, ChromaDB, SQL, Pandas, Docker.
"""
    },
    {
        "id": "cand_03",
        "category": "Java Developer",
        "raw_resume": """
Robert Brown - Backend Engineer (rbrown@enterprise.org, +44 20 7946 0912)
Experience:
- Built resilient distributed microservices using Java 17, Spring Boot, and Apache Kafka.
- Designed relational schemas and managed migration scripts across PostgreSQL instances.
- Engineered high-throughput transaction processing systems handling 5,000 requests/sec.
Skills: Java, Spring Boot, Kafka, PostgreSQL, Docker, Kubernetes, AWS.
"""
    }
]

def main():
    parser = argparse.ArgumentParser(description="AI Expert Finder Query Demo")
    parser.add_argument("--query", type=str, default="Frontend Engineer with strong React and TypeScript experience",
                        help="Target job requirement or skill description")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top candidates to return")
    args = parser.parse_args()

    print("=" * 60)
    print("AI EXPERT FINDER: RECRUITMENT MATCHING PIPELINE DEMO")
    print("=" * 60)
    print(f"\n[Target Query]: \"{args.query}\"\n")

    preprocessor = ResumePreprocessor(mask_pii=True)
    re_ranker = LLMReRanker()

    print("[Step 1] Preprocessing & Masking PII...")
    processed_candidates = []
    for cand in SAMPLE_CANDIDATES:
        processed_text = preprocessor.process(cand["raw_resume"])
        processed_candidates.append({
            "id": cand["id"],
            "category": cand["category"],
            "text": processed_text
        })
    print(f"Processed {len(processed_candidates)} candidate profiles.\n")

    print("[Step 2] Computing Relevance & Ranking...")
    # Simple term overlap & similarity heuristic for offline zero-dependency demo
    query_terms = set(args.query.lower().replace(",", "").split())
    for cand in processed_candidates:
        words = set(cand["text"].lower().split())
        overlap = len(query_terms.intersection(words))
        cand["distance"] = max(0.05, 1.0 - (overlap / (len(query_terms) + 1e-5)))

    ranked = re_ranker.rank_candidates(args.query, processed_candidates)

    print(f"[Step 3] Top {min(args.top_k, len(ranked))} Matches Identified:\n")
    for rank, cand in enumerate(ranked[:args.top_k], start=1):
        print(f"Rank {rank}: [{cand['category']}] (Candidate ID: {cand['id']})")
        print(f"  Relevance Score : {cand['relevance_score']}/10")
        print(f"  Summary         : {cand['eval_summary']}")
        print(f"  Masked Profile  : {cand['text'].strip()[:180]}...")
        print("-" * 60)

if __name__ == "__main__":
    main()
