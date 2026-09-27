#!/usr/bin/env python3
"""
Semantic Query Router - Categorizes user input to optimize compute allocation.
Routes incoming queries into:
- 'general_chat': Greetings, pleasantries, small talk (direct LLM response without vector search)
- 'codebase_ast': Questions about repo architecture, files, callers/callees (handled via AST graph)
- 'vector_search': Technical domain questions, documentation lookups (handled via ZeroVRAMRetriever)
"""

import os
import re
from typing import Optional, Any

try:
    import ollama
    from ollama import Client
except ImportError:
    ollama = None
    Client = None

# Fast heuristic regex patterns for zero-token classification
GREETINGS_PATTERN = re.compile(
    r"\b(hi|hello|hey|good\s+(morning|afternoon|evening|day)|howdy|greetings|hola|saludos|thanks|thank\s+you|bye|goodbye|who\s+are\s+you)\b",
    re.IGNORECASE
)

CODEBASE_AST_PATTERN = re.compile(
    r"\b(codebase|repository|repo\s+graph|ast|callee|caller|dependencies|call\s+graph|local_codegraph|map_repository|architecture\s+map|file\s+structure)\b",
    re.IGNORECASE
)

TECHNICAL_KEYWORDS = re.compile(
    r"\b(vram|qdrant|embedding|fastembed|flashrank|rerank|docker|model|parameter|chunk|gpu|cuda|cpu|vector|chroma|ollama|rag)\b",
    re.IGNORECASE
)


def route_query_heuristic(query: str) -> Optional[str]:
    """
    Fast-path heuristic routing. Returns category or None if ambiguous.
    """
    cleaned = query.strip()
    if not cleaned:
        return "general_chat"

    if CODEBASE_AST_PATTERN.search(cleaned):
        return "codebase_ast"

    if GREETINGS_PATTERN.search(cleaned) and not TECHNICAL_KEYWORDS.search(cleaned):
        return "general_chat"

    if TECHNICAL_KEYWORDS.search(cleaned):
        return "vector_search"

    return None


    return None


def route_query(
    query: str,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    client: Optional[Any] = None
) -> str:
    """
    Determines optimal execution route for the user query.
    Falls back to LLM classifier when heuristics are ambiguous.
    """
    # 1. Heuristic fast path
    heuristic_route = route_query_heuristic(query)
    if heuristic_route:
        return heuristic_route

    # 2. LLM classifier fallback
    base_url = base_url or os.getenv("BASE_URL", "http://ollama:11434")
    model_name = model_name or os.getenv("OLLAMA_ROUTER_MODEL", "qwen2.5:7b-fenced")

    classification_prompt = (
        "You are an expert query router. Classify the following user query into exactly ONE category:\n"
        "- general_chat: conversational greetings, small talk, pleasantries.\n"
        "- codebase_ast: questions specifically about this codebase structure, functions, classes, dependencies, or repo files.\n"
        "- vector_search: technical questions, concepts, machine learning, data science, documentation, or PDF content.\n\n"
        "Query: \"\"\"{query}\"\"\"\n\n"
        "Return ONLY the category name ('general_chat', 'codebase_ast', or 'vector_search') with no other text or explanation."
    ).format(query=query)

    try:
        if client is None and Client is not None:
            client = Client(host=base_url, timeout=5.0)

        if client is not None:
            response = client.chat(
                model=model_name,
                messages=[{"role": "user", "content": classification_prompt}],
                options={"temperature": 0.0, "num_ctx": 1024}
            )
            raw_text = (
                response.get("message", {}).get("content", "")
                if isinstance(response, dict)
                else getattr(getattr(response, "message", None), "content", "")
            )
            raw_text = raw_text.strip().lower()

            if "general_chat" in raw_text:
                return "general_chat"
            elif "codebase_ast" in raw_text:
                return "codebase_ast"
            elif "vector_search" in raw_text:
                return "vector_search"
    except Exception as e:
        print(f"⚠️ Router LLM classification fallback error ({type(e).__name__}: {e}) - defaulting to vector_search")

    # Default safe fallback for substantive queries
    return "vector_search"
