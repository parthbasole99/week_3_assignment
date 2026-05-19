#!/usr/bin/env python3
"""
RAG Generation Pipeline for SIGGRAPH 2025 Papers.

Uses the retrieval pipeline to find relevant chunks,
then generates an answer using an LLM via OpenRouter API.

Usage:
    from rag_generate import RAGGenerator, GenerationConfig, SYSTEM_PROMPT

    generator = RAGGenerator()
    result = generator.generate("What is 3D Gaussian Splatting?")
    print(result["answer"])
"""

import os
import requests
from typing import Optional
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

from retrieval_pipeline import RetrievalPipeline, RetrievalResult


SYSTEM_PROMPT = """You are an expert research assistant specializing in computer graphics, specifically SIGGRAPH 2025 papers.

Your task is to answer questions using ONLY the provided research paper excerpts.

Rules:
1. Cite sources using [Paper Title] format
2. Be comprehensive and technically accurate
3. If the excerpts don't contain the answer, say so
4. Use LaTeX for math: $inline$ or $$block$$
5. Do NOT make up information not in the excerpts
6. Do NOT include a References section at the end
"""


QUERY_REFINEMENT_PROMPT = """You are an expert at refining search queries for academic paper retrieval.

Given a user's question, rewrite it as a clear, focused search query that will retrieve the most relevant research papers.

Keep it concise (under 20 words). Focus on key technical terms.

User question: {query}

Refined search query:"""


@dataclass
class GenerationConfig:
    llm_model: str = "openai/gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 2000
    openrouter_api_key: Optional[str] = None
    refine_query: bool = True
    refinement_model: str = "openai/gpt-3.5-turbo"
    retrieval_top_k: int = 8


class RAGGenerator:
    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        retrieval_pipeline=None,
    ):
        self.config = config or GenerationConfig()
        self.retrieval = retrieval_pipeline or RetrievalPipeline()

        self.openrouter_api_key = (
            self.config.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        )
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        self.openrouter_base_url = "https://openrouter.ai/api/v1"

    def refine_query(self, query: str) -> str:
        if not self.config.refine_query:
            return query

        prompt = QUERY_REFINEMENT_PROMPT.format(query=query)

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.refinement_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 100,
        }

        try:
            response = requests.post(
                f"{self.openrouter_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            if response.status_code != 200:
                print(f"Query refinement failed ({response.status_code}): {response.text}")
                return query
            data = response.json()
            refined = data["choices"][0]["message"]["content"].strip()
            return refined.strip('"').strip("'") or query
        except Exception as e:
            print(f"Query refinement error: {e}")
            return query

    def _format_context(self, results: list[RetrievalResult]) -> str:
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"--- Source {i} ---\n"
                f"Title: {r.title}\n"
                f"Authors: {r.authors}\n"
                f"Section: {r.chunk_section}\n"
                f"\n"
                f"Content:\n"
                f"{r.text}\n"
            )
        return "\n".join(parts)

    def _build_sources_metadata(self, results: list[RetrievalResult]) -> list[dict]:
        seen: dict[str, dict] = {}
        for r in results:
            if r.title not in seen:
                seen[r.title] = {
                    "title": r.title,
                    "authors": r.authors,
                    "pdf_url": r.pdf_url,
                    "github_link": r.github_link,
                    "video_link": r.video_link,
                    "acm_url": r.acm_url,
                    "abstract_url": r.abstract_url,
                }
        return list(seen.values())

    def _call_llm(self, query: str, context: str) -> str:
        user_message = (
            f"Based on the following research paper excerpts, answer this question.\n\n"
            f"Question: {query}\n\n"
            f"Research Paper Excerpts:\n{context}\n\n"
            f"Remember to cite papers using [Paper Title] format."
        )

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.llm_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        response = requests.post(
            f"{self.openrouter_base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"LLM API failed ({response.status_code}): {response.text}"
            )
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def generate(
        self,
        query: str,
        top_k: Optional[int] = None,
        return_sources: bool = True,
    ) -> dict:
        refined = self.refine_query(query)

        k = top_k if top_k is not None else self.config.retrieval_top_k
        results = self.retrieval.retrieve(refined, top_k=k)

        if not results:
            return {
                "query": query,
                "refined_query": refined,
                "answer": "I couldn't find any relevant papers to answer this question.",
                "sources": [],
            }

        context = self._format_context(results)
        answer = self._call_llm(query, context)

        return {
            "query": query,
            "refined_query": refined,
            "answer": answer,
            "sources": self._build_sources_metadata(results) if return_sources else [],
        }


if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "What is 3D Gaussian Splatting?"

    print("Initializing RAG Generator...")
    generator = RAGGenerator()

    print(f"\nQuery: {query}")
    print("=" * 60)

    result = generator.generate(query)

    print(f"Refined Query: {result.get('refined_query', 'N/A')}")
    print("=" * 60)
    print("\nAnswer:")
    print(result["answer"])
    print("=" * 60)
    print(f"\nSources: {len(result.get('sources', []))} papers")
    for source in result.get("sources", []):
        print(f"  - {source['title']}")
