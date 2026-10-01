#!/usr/bin/env python3
"""
Protocols — Structural Typing Contracts for SOLID Compliance.

Defines abstract interfaces decoupling CRAG graph, factory,
and retrieval layer from concrete provider implementations (Local Ollama,
GCP Vertex AI, etc.).

SOLID Principles enforced:
  - LSP (Liskov Substitution): Any conforming implementation can replace another.
  - DIP (Dependency Inversion): Callers depend on protocols, not concretions.
  - ISP (Interface Segregation): Focused interfaces instead of mega-class.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


# ─────────────────────────────────────────────────────────────────────────────
# SOLID: Interface Segregation Principle
# RetrieverProtocol and LLMProviderProtocol are intentionally separate.
# Grader never imports LLMProviderProtocol.
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class RetrieverProtocol(Protocol):
    """
    Structural interface for vector retrieval backends.

    SOLID — LSP Guarantee:
        ZeroVRAMRetriever and GCP Vertex AI retriever conform to this interface.
        CRAG graph nodes receive RetrieverProtocol without knowing concrete backend.

    Contract:
        retrieve_and_rerank() MUST return List[Dict[str, Any]] containing
        at minimum keys 'text' (str) and 'score' (float).
    """

    def retrieve_and_rerank(
        self,
        query: str,
        retrieve_limit: int = 50,
        rerank_top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve candidates and return reranked top-k passages."""
        ...


@runtime_checkable
class LLMProviderProtocol(Protocol):
    """
    Structural interface for LLM generation backends.

    SOLID — DIP Guarantee:
        CRAG graph nodes receive LLMProviderProtocol instance at build time
        via functools.partial injection. Never import ollama or Vertex AI directly.

    Contract:
        generate() MUST return non-empty str.
        provider_type MUST be identifier string.
    """

    provider_type: str

    def generate(self, prompt: str, context: str = "") -> str:
        """Generate response given prompt and optional context string."""
        ...

    def __call__(self, prompt: str, context: str = "") -> str:
        """Allow provider callable syntax."""
        ...
