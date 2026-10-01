#!/usr/bin/env python3
"""
Retrieval Grader & Query Rewriter (CRAG - Corrective RAG).
Optimized for Zero-VRAM architecture:
- Batch grading: ALL documents graded in ONE LLM call (eliminates model reloads)
- Unified num_ctx: Matches generator context (4096) to prevent Ollama KV cache rebuilds
- Heuristic pre-filter: Skips LLM grading when vector scores are already >= 0.85
"""

import os
import json
import re
from typing import Optional, Any, List, Dict, Tuple

try:
    import ollama
    from ollama import Client
except ImportError:
    ollama = None
    Client = None

# Unified context size — MUST match OLLAMA_NUM_CTX in docker-compose.yaml
# to prevent Ollama from reloading the model between grader/generator calls
UNIFIED_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

# Heuristic threshold: if top vector score exceeds this, skip LLM grading entirely
HEURISTIC_SKIP_THRESHOLD = float(os.getenv("GRADER_HEURISTIC_THRESHOLD", "0.85"))

# Max documents to include in a single batch grading prompt
MAX_BATCH_DOCS = 10


def _heuristic_relevance(query: str, doc_text: str) -> float:
    """
    Zero-cost lexical overlap heuristic.
    Returns a score between 0.0 and 1.0 based on token intersection.
    Used as a pre-filter to avoid expensive LLM grading calls.
    """
    if not doc_text or not doc_text.strip():
        return 0.0

    query_tokens = set(re.findall(r'\b[a-zA-Z0-9_]{3,}\b', query.lower()))
    doc_tokens = set(re.findall(r'\b[a-zA-Z0-9_]{3,}\b', doc_text.lower()))

    if not query_tokens:
        return 0.5  # Neutral for empty queries

    overlap = query_tokens.intersection(doc_tokens)
    return len(overlap) / len(query_tokens)


def _parse_batch_verdicts(raw: str, expected_count: int) -> Optional[List[bool]]:
    """Robust parser for LLM batch relevance verdicts."""
    cleaned = raw.strip()
    cleaned = re.sub(r'```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'```\s*$', '', cleaned)

    match = re.search(r'\[[\s\S]*?\]', cleaned)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                verdicts = []
                for item in parsed:
                    if isinstance(item, bool):
                        verdicts.append(item)
                    elif isinstance(item, str):
                        verdicts.append(item.lower() in ("true", "yes", "relevant", "1"))
                    elif isinstance(item, (int, float)):
                        verdicts.append(bool(item))
                if len(verdicts) == expected_count:
                    return verdicts
                elif len(verdicts) > 0:
                    if len(verdicts) < expected_count:
                        verdicts.extend([True] * (expected_count - len(verdicts)))
                    return verdicts[:expected_count]
        except Exception:
            pass

    if expected_count == 1:
        low = cleaned.lower()
        if "yes" in low or "true" in low:
            return [True]
        if "no" in low or "false" in low:
            return [False]

    tokens = re.findall(r'\b(true|false|yes|no)\b', cleaned, re.IGNORECASE)
    if len(tokens) == expected_count:
        return [t.lower() in ("true", "yes") for t in tokens]

    return None


def _heuristic_fallback(
    query: str,
    documents: List[Dict[str, Any]]
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Zero-cost fallback when LLM grading is unavailable or fails.
    Uses lexical token overlap combined with vector similarity to score relevance.
    """
    scored = []
    for doc in documents:
        text = doc.get("text", "")
        h_score = _heuristic_relevance(query, text)
        raw_score = doc.get("score")
        score_val = float(raw_score) if raw_score is not None else 0.0
        combined = 0.6 * score_val + 0.4 * h_score
        scored.append((combined, doc))

    scored.sort(key=lambda x: -x[0])
    relevant = [doc for score, doc in scored if score > 0.3]
    is_overall = len(relevant) > 0
    print(f"ℹ️ Grader: Heuristic fallback. {len(relevant)}/{len(documents)} docs above threshold.")
    return is_overall, relevant


def grade_documents_batched(
    query: str,
    documents: List[Dict[str, Any]],
    client: Optional[Any] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    BATCHED grading: Evaluates ALL documents in a SINGLE LLM call.
    Eliminates the N×model-reload catastrophe of per-document grading.

    Strategy:
    1. If top vector score >= HEURISTIC_SKIP_THRESHOLD → accept all, skip LLM
    2. Otherwise → send all docs in one prompt, get structured yes/no per doc
    3. On LLM failure → fall back to heuristic lexical scoring
    """
    if not documents:
        return False, []

    # Guard: retrieval fallback / error context
    if len(documents) == 1 and documents[0].get("id") == "fallback":
        return False, []

    base_url = base_url or os.getenv("BASE_URL", "http://ollama:11434")
    model_name = model_name or os.getenv("OLLAMA_GRADER_MODEL", "qwen2.5:7b-fenced")

    # ── PHASE 1: Heuristic Pre-Filter ──────────────────────────────────
    # If the vector search already returned high-confidence results,
    # skip the LLM grader entirely (zero token cost, zero latency)
    top_score = max((float(d.get("score") or 0.0) for d in documents), default=0.0)
    if top_score >= HEURISTIC_SKIP_THRESHOLD:
        print(f"✅ Grader: Heuristic bypass (top_score={top_score:.4f} >= {HEURISTIC_SKIP_THRESHOLD}). Accepting all {len(documents)} docs.")
        return True, documents

    # ── PHASE 2: Batched LLM Grading (ONE call for ALL docs) ──────────
    docs_to_grade = documents[:MAX_BATCH_DOCS]

    doc_sections = []
    for i, doc in enumerate(docs_to_grade):
        text = (doc.get("text") or "")[:2000]
        doc_sections.append(f"[DOC {i}] {text}")

    all_docs_text = "\n\n".join(doc_sections)

    batch_prompt = (
        f"You are an objective relevance evaluator. Below are {len(docs_to_grade)} retrieved document excerpts "
        f"and a user question. For EACH document, determine if it contains information relevant to answering the question.\n\n"
        f"User Question: \"{query}\"\n\n"
        f"Documents:\n{all_docs_text}\n\n"
        f"For each document, output a JSON array of booleans indicating relevance.\n"
        f"Example: [true, false, true, true, false]\n"
        f"Output ONLY the JSON array. No explanation."
    )

    try:
        if client is None and Client is not None:
            client = Client(host=base_url, timeout=30.0)

        if client is not None:
            response = client.chat(
                model=model_name,
                messages=[{"role": "user", "content": batch_prompt}],
                options={
                    "temperature": 0.0,
                    "num_ctx": UNIFIED_NUM_CTX,
                },
            )
            raw = (
                response.get("message", {}).get("content", "")
                if isinstance(response, dict)
                else getattr(getattr(response, "message", None), "content", "")
            ).strip()

            verdicts = _parse_batch_verdicts(raw, len(docs_to_grade))
            if verdicts is not None:
                relevant_docs = [
                    doc for doc, is_rel in zip(docs_to_grade, verdicts) if is_rel
                ]
                relevant_docs.extend(documents[MAX_BATCH_DOCS:])
                is_overall = len(relevant_docs) > 0
                print(f"✅ Grader: Batched LLM grading complete. {len(relevant_docs)}/{len(documents)} relevant.")
                return is_overall, relevant_docs

            print(f"⚠️ Grader: Batched LLM response unparseable. Falling back to heuristic.")
            return _heuristic_fallback(query, documents)

    except Exception as e:
        print(f"⚠️ Grader: Batched LLM call failed ({type(e).__name__}: {e}). Falling back to heuristic.")
        return _heuristic_fallback(query, documents)

    return _heuristic_fallback(query, documents)


def grade_single_document(
    query: str,
    doc_text: str,
    client: Optional[Any] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None
) -> bool:
    """
    Backwards-compatible helper for grading a single document.
    """
    if not doc_text or not doc_text.strip():
        return False
    is_rel, _ = grade_documents_batched(
        query=query,
        documents=[{"id": 0, "text": doc_text, "score": 0.0}],
        client=client,
        model_name=model_name,
        base_url=base_url
    )
    return is_rel


def grade_documents(
    query: str,
    documents: List[Dict[str, Any]],
    client: Optional[Any] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Public API preserved for run_rag.py and test suite compatibility.
    Delegates to high-throughput batched implementation.
    """
    return grade_documents_batched(
        query=query,
        documents=documents,
        client=client,
        model_name=model_name,
        base_url=base_url
    )


def rewrite_query(
    query: str,
    client: Optional[Any] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None
) -> str:
    """
    Rephrases user query to optimize semantic retrieval from vector database.
    Uses UNIFIED_NUM_CTX to prevent model reload.
    """
    base_url = base_url or os.getenv("BASE_URL", "http://ollama:11434")
    model_name = model_name or os.getenv("OLLAMA_REWRITER_MODEL", "qwen2.5:7b-fenced")

    rewrite_prompt = (
        "You are an expert query optimizer for dense and sparse vector retrieval.\n"
        "Analyze the following user question and reformulate it into an effective, clear, keyword-rich search query.\n\n"
        f"Original Question: \"\"\"{query}\"\"\"\n\n"
        "Provide ONLY the rewritten search query. Do not include quotes, preamble, or explanation."
    )

    try:
        if client is None and Client is not None:
            client = Client(host=base_url, timeout=30.0)

        if client is not None:
            response = client.chat(
                model=model_name,
                messages=[{"role": "user", "content": rewrite_prompt}],
                options={
                    "temperature": 0.2,
                    "num_ctx": UNIFIED_NUM_CTX,
                }
            )
            raw = (
                response.get("message", {}).get("content", "")
                if isinstance(response, dict)
                else getattr(getattr(response, "message", None), "content", "")
            ).strip()

            cleaned = re.sub(r"^[\"']|[\"']$", "", raw).strip()
            return cleaned if cleaned else query
    except Exception as e:
        print(f"⚠️ Query rewrite error ({type(e).__name__}: {e}) - keeping original query")

    return query
