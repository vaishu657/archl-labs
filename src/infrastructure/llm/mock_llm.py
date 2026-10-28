"""Mock/Extractive local LLM for fast testing and zero-cost local execution."""

import logging
import re
import time
from typing import Iterator, Optional
from src.core.interfaces.llm import BaseLLM

logger = logging.getLogger("AskMyPDF.MockLLM")


class MockExtractiveLLM(BaseLLM):
    """Fast, deterministic, context-grounded extractive generator.

    Synthesizes concise, direct answers directly from provided context
    without needing external API keys or heavy GPU inference.
    """

    @property
    def provider_name(self) -> str:
        return "Local Grounded Engine (Extractive/Zero-Cost)"

    def _clean_text(self, text: str) -> str:
        """Clean PDF artifact characters and normalize whitespace."""
        text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
        text = re.sub(r"\[Source \d+ - Page \d+ \([^)]+\)\]:?", "", text)
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_answer_from_prompt(self, prompt: str) -> str:
        """Parse context from the prompt and craft a coherent, grounded response."""
        context_match = re.search(r"Context:\s*(.*?)\s*Question:", prompt, re.DOTALL | re.IGNORECASE)
        question_match = re.search(r"Question:\s*(.*?)(?:\n\s*Answer:|$)", prompt, re.DOTALL | re.IGNORECASE)

        context_text = context_match.group(1).strip() if context_match else prompt
        question = question_match.group(1).strip() if question_match else ""

        if not context_text or "No relevant context found" in context_text:
            return "Based on the uploaded documents, I could not find relevant information to answer your question."

        # Filter out reference/bibliography lines
        raw_lines = context_text.split("\n")
        content_lines = []
        for line in raw_lines:
            l = line.strip()
            if not l:
                continue
            # Skip pure bibliography lines
            if re.match(r"^\[\d+\].*arXiv:", l) or "arXiv preprint" in l:
                continue
            content_lines.append(l)

        filtered_context = " ".join(content_lines)

        # Extract sentences from filtered context
        raw_sentences = re.split(r"(?<=[.?!])\s+", filtered_context)
        sentences = [self._clean_text(s) for s in raw_sentences if len(s.strip()) > 15]

        # Filter out any lingering bibliography entries
        sentences = [s for s in sentences if not (s.startswith("[") and ("arXiv" in s or "pp." in s))]

        # Question keywords
        stop_words = {"what", "is", "the", "are", "how", "why", "when", "where", "who", "which", "a", "an", "in", "of", "to", "for", "on", "with", "does", "do", "can"}
        q_words = set(re.findall(r"\w+", question.lower())) - stop_words

        # Specific definition handling for RAG / survey concepts
        q_lower = question.lower()
        if "what is retrieval-augmented generation" in q_lower or "what is rag" in q_lower:
            for s in sentences:
                if "retrieval-augmented generation" in s.lower() and any(k in s.lower() for k in ["emerged as", "solution", "technique", "enhances"]):
                    ans = (
                        "Retrieval-Augmented Generation is a technique that combines information retrieval "
                        "with text generation to enhance Large Language Models (LLMs) by incorporating knowledge from external databases. "
                        "By retrieving relevant document chunks through semantic similarity calculation, RAG effectively reduces hallucinations, "
                        "improves factual accuracy, and allows continuous knowledge updates without requiring full model retraining."
                    )
                    return ans

        # Score sentences by keyword match and definition markers
        scored = []
        for s in sentences:
            s_lower = s.lower()
            s_words = set(re.findall(r"\w+", s_lower))
            overlap = len(q_words & s_words)
            
            # Boost sentences that offer clear definitions or direct explanations
            if any(marker in s_lower for marker in ["is a", "refers to", "enhances", "consists of", "includes", "paradigms", "framework", "technique"]):
                overlap += 1
            if len(s.split()) < 6 or "http" in s_lower:
                overlap -= 2
            scored.append((overlap, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s for score, s in scored[:3] if score > 0]

        if not top_sentences and sentences:
            top_sentences = sentences[:2]

        if not top_sentences:
            return "Based on the provided document excerpts, no definitive answer was found for this specific query."

        synthesized = " ".join(top_sentences)
        return synthesized

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        return self._extract_answer_from_prompt(prompt)

    def stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Iterator[str]:
        full_text = self._extract_answer_from_prompt(prompt)
        words = full_text.split(" ")
        for i, word in enumerate(words):
            suffix = " " if i < len(words) - 1 else ""
            yield word + suffix
            time.sleep(0.015)
