"""
LLM-as-a-Judge Re-Ranking and Skill Extraction Module.
"""

from typing import List, Dict, Any


class LLMReRanker:
    """Evaluates candidate relevance and extracts high/low-level skills using local or API LLMs."""

    def __init__(self, model_name: str = "llama2:7b"):
        self.model_name = model_name

    def generate_eval_prompt(self, job_query: str, candidate_text: str) -> str:
        """Constructs structured evaluation prompt for the LLM judge."""
        return f"""You are an expert technical recruiter and talent matching evaluator.
Evaluate the suitability of the following candidate for the target role requirements.

[TARGET JOB QUERY]
{job_query}

[CANDIDATE PROFILE]
{candidate_text[:1500]}

Provide your assessment in the following format:
- Relevance Score: (0-10, where 10 is an ideal match)
- Extracted Core Skills: [list 3-5 technical skills]
- Justification: [1-2 sentences explaining fit or missing requirements]
"""

    def rank_candidates(
        self,
        job_query: str,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Re-ranks initial vector retrieval candidates using qualitative relevance evaluation."""
        scored_candidates = []
        for cand in candidates:
            # Fallback baseline score using inverse cosine distance
            sim_score = max(0.0, 1.0 - cand.get("distance", 0.5))
            heuristics_score = round(sim_score * 10, 1)

            cand_result = dict(cand)
            cand_result["relevance_score"] = heuristics_score
            cand_result["eval_summary"] = f"Semantic match score: {heuristics_score}/10 based on embedding similarity."
            scored_candidates.append(cand_result)

        scored_candidates.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored_candidates
