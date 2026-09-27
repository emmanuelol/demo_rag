#!/usr/bin/env python3
"""
Retrieval Grader & Query Rewriter (CRAG - Corrective RAG).
Evaluates retrieved context relevance against the user query.
Filters out irrelevant passages and rewrites search queries when relevance threshold fails.
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


def grade_single_document(
    query: str,
    doc_text: str,
    client: Optional[Any] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None
) -> bool:
    """
    Evaluates whether a single retrieved document text is relevant to the question.
    Returns True if relevant ('yes'), False otherwise.
    """
    if not doc_text or not doc_text.strip():
        return False

    base_url = base_url or os.getenv("BASE_URL", "http://ollama:11434")
    model_name = model_name or os.getenv("OLLAMA_GRADER_MODEL", "qwen2.5:7b-fenced")

    grading_prompt = (
        "You are an objective evaluator assessing whether a retrieved document contains relevant information to answer a user question.\n\n"
        "Retrieved Document:\n\"\"\"{doc_text}\"\"\"\n\n"
        "User Question:\n\"\"\"{query}\"\"\"\n\n"
        "Give a binary score 'yes' or 'no' indicating whether the document contains information relevant to the question.\n"
        "Answer with ONLY 'yes' or 'no'."
    ).format(doc_text=doc_text[:1500], query=query)

    try:
        if client is None and Client is not None:
            client = Client(host=base_url, timeout=5.0)

        if client is not None:
            response = client.chat(
                model=model_name,
                messages=[{"role": "user", "content": grading_prompt}],
                options={"temperature": 0.0, "num_ctx": 2048}
            )
            raw = (
                response.get("message", {}).get("content", "")
                if isinstance(response, dict)
                else getattr(getattr(response, "message", None), "content", "")
            ).strip().lower()

            return "yes" in raw
    except Exception as e:
        print(f"⚠️ Document grading error ({type(e).__name__}: {e}) - permissive fallback")
        return True

    return True


def grade_documents(
    query: str,
    documents: List[Dict[str, Any]],
    client: Optional[Any] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Grades retrieved documents.
    Returns (is_relevant, filtered_documents).
    If no documents are relevant, returns (False, []).
    """
    if not documents:
        return False, []

    # Check if retrieval returned fallback / error context
    if len(documents) == 1 and documents[0].get("id") == "fallback":
        return False, []

    relevant_docs = []
    for doc in documents:
        text = doc.get("text", "")
        # Empty text is non-relevant
        if not text or not text.strip():
            continue

        is_relevant = grade_single_document(
            query=query,
            doc_text=text,
            client=client,
            model_name=model_name,
            base_url=base_url
        )
        if is_relevant:
            relevant_docs.append(doc)

    is_overall_relevant = len(relevant_docs) > 0
    return is_overall_relevant, relevant_docs


def rewrite_query(
    query: str,
    client: Optional[Any] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None
) -> str:
    """
    Rephrases the user query to optimize semantic retrieval from vector database.
    """
    base_url = base_url or os.getenv("BASE_URL", "http://ollama:11434")
    model_name = model_name or os.getenv("OLLAMA_REWRITER_MODEL", "qwen2.5:7b-fenced")

    rewrite_prompt = (
        "You are an expert query optimizer for dense and sparse vector retrieval.\n"
        "Analyze the following user question and reformulate it into an effective, clear, keyword-rich search query.\n\n"
        "Original Question: \"\"\"{query}\"\"\"\n\n"
        "Provide ONLY the rewritten search query. Do not include quotes, preamble, or explanation."
    ).format(query=query)

    try:
        if client is None and Client is not None:
            client = Client(host=base_url, timeout=5.0)

        if client is not None:
            response = client.chat(
                model=model_name,
                messages=[{"role": "user", "content": rewrite_prompt}],
                options={"temperature": 0.2, "num_ctx": 1024}
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
