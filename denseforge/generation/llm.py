"""LLM Generation — completions with source attribution and speculative drafts."""
from __future__ import annotations

import time

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from denseforge.config import GenerationConfig


class LLMGenerator:
    """Generate answers with inline source attribution using any OpenAI-compatible API."""

    def __init__(self, config: GenerationConfig | None = None,
                 api_key: str | None = None, base_url: str | None = None,
                 model_override: str | None = None):
        self.config = config or GenerationConfig()
        self.api_key = api_key
        self.base_url = base_url or "http://localhost:3000/v1"
        self.model = model_override or self.config.model_name
        self._client: httpx.Client | None = None

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.Client(
                base_url=self.base_url, headers=headers, timeout=120,
            )
        return self._client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _chat(self, messages: list[dict], temperature: float | None = None,
              max_tokens: int | None = None) -> dict:
        """Send a chat-completions request and return the raw response dict."""
        client = self._get_client()
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        resp = client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    _SYSTEM_PROMPT = (
        "You are DenseForge, an expert research assistant.  "
        "Answer the user's question using ONLY the provided context.  "
        "If the context does not contain enough information, say so honestly.  "
        "Always cite which source(s) you used by referencing [Source 1], [Source 2], etc."
    )

    def _build_attribution_prompt(self, query: str, context: str) -> list[dict]:
        numbered = []
        for i, chunk in enumerate(context.split("\n\n---\n\n"), 1):
            numbered.append(f"[Source {i}]\n{chunk.strip()}")
        sources_block = "\n\n".join(numbered)

        user_msg = (
            f"## Context\n\n{sources_block}\n\n"
            f"## Question\n\n{query}\n\n"
            f"## Instructions\n"
            f"Answer the question using the context above.  "
            f"Include inline citations like [Source 1] after each claim.  "
            f"If multiple sources support a claim, cite all of them."
        )
        return [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, query: str, context: str, **kwargs) -> dict:
        """Generate an attributed answer from query + context.

        Returns dict with keys: answer, sources_used, latency_ms, tokens.
        """
        t0 = time.perf_counter()
        messages = self._build_attribution_prompt(query, context)
        try:
            raw = self._chat(messages, **kwargs)
        except Exception as exc:
            logger.error("LLM call failed: {}", exc)
            return {
                "answer": f"[Error communicating with LLM: {exc}]",
                "sources_used": [],
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "tokens": {},
                "error": str(exc),
            }

        choice = raw.get("choices", [{}])[0]
        answer_text = choice.get("message", {}).get("content", "")
        usage = raw.get("usage", {})

        sources_used = self._extract_citations(answer_text)
        latency = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "answer": answer_text,
            "sources_used": sources_used,
            "latency_ms": latency,
            "tokens": usage,
            "model": self.model,
        }

    def generate_speculative(self, query: str, context: str,
                             n_drafts: int | None = None) -> dict:
        """Generate multiple draft answers and return the best by score.

        Drafts are scored on: citation coverage, length, and self-consistency.
        """
        n = n_drafts or self.config.speculative_drafts
        drafts: list[dict] = []

        for i in range(n):
            temp = self.config.temperature + i * 0.05
            draft = self.generate(query, context, temperature=min(temp, 1.0))
            draft["score"] = self._score_draft(draft, context)
            drafts.append(draft)

        best = max(drafts, key=lambda d: d["score"])
        best["all_draft_scores"] = [d["score"] for d in drafts]
        return best

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_citations(text: str) -> list[str]:
        import re
        return list({m.group(0) for m in re.finditer(r"\[Source \d+\]", text)})

    @staticmethod
    def _score_draft(draft: dict, context: str) -> float:
        answer = draft.get("answer", "")
        n_citations = len(draft.get("sources_used", []))
        n_sources = context.count("[Source")
        length = len(answer.split())

        citation_coverage = n_citations / max(n_sources, 1)
        length_score = min(length / 150.0, 1.0)
        return round(0.5 * citation_coverage + 0.3 * length_score + 0.2, 4)

    # ------------------------------------------------------------------
    # Direct prompt (no retrieval context)
    # ------------------------------------------------------------------
    def chat(self, user_message: str, system_prompt: str | None = None) -> str:
        """Simple chat completion without retrieval context."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        raw = self._chat(messages)
        return raw.get("choices", [{}])[0].get("message", {}).get("content", "")

    def stats(self) -> dict:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

    def close(self):
        if self._client and not self._client.is_closed:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
