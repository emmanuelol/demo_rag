#!/usr/bin/env python3
"""
scripts/crag_graph.py — Corrective RAG (CRAG) State Machine.

SOLID Principles:
  - SRP (Single Responsibility): Manages CRAG state machine, routing, and transitions.
  - DIP (Dependency Inversion): Injects LLMProviderProtocol, RetrieverProtocol, and scoring callbacks.
  - LSP (Liskov Substitution): Works with any conforming retriever or LLM implementation.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    StateGraph = None
    END = None

from scripts.protocols import RetrieverProtocol, LLMProviderProtocol
from scripts.router import route_query as default_route_query
from scripts.grader import grade_documents as default_grade_documents, rewrite_query as default_rewrite_query
from scripts.llm_utils import ollama_llm


class AgentState(TypedDict):
    """
    CRAG Agent State schema.
    Tracks query, routing category, retrieved documents, relevance evaluation,
    generation payload, retry counters, and full audit trace.
    """
    question: str
    search_query: str
    route: str
    documents: List[Dict[str, Any]]
    is_relevant: bool
    generation: str
    retry_count: int
    max_retries: int
    trace: List[str]


# ─────────────────────────────────────────────────────────────
# SOLID: Dependency Inversion — Node closures accept abstractions
# ─────────────────────────────────────────────────────────────


def route_node(
    state: AgentState,
    router_fn: Optional[Callable] = None,
    model_embedding: str = "qwen2.5:7b-fenced"
) -> dict:
    _route = router_fn or default_route_query
    route = _route(state["question"], model_name=model_embedding)
    return {"route": route, "trace": state.get("trace", []) + [f"route:{route}"]}


def direct_chat_node(
    state: AgentState,
    llm_provider: Optional[LLMProviderProtocol] = None,
    llm_fn: Optional[Callable] = None,
    model_embedding: str = "qwen2.5:7b-fenced"
) -> dict:
    if llm_provider is not None:
        ans = llm_provider.generate(state["question"], context="")
    elif llm_fn is not None:
        ans = llm_fn(state["question"], "", model_embedding)
    else:
        ans = ollama_llm(state["question"], context="", model_embedding=model_embedding)
    return {"generation": ans, "trace": state.get("trace", []) + ["direct_chat"]}


def codebase_ast_node(
    state: AgentState,
    llm_provider: Optional[LLMProviderProtocol] = None,
    llm_fn: Optional[Callable] = None,
    model_embedding: str = "qwen2.5:7b-fenced"
) -> dict:
    summary_ctx = (
        "Repository architecture comprises modular services: "
        "Semantic Router (scripts/router.py), Retrieval Grader (scripts/grader.py), "
        "ZeroVRAMRetriever (scripts/context_assembler.py), Provider Factory (scripts/factory.py), "
        "and LangGraph State Machine (scripts/crag_graph.py)."
    )
    if llm_provider is not None:
        ans = llm_provider.generate(state["question"], context=summary_ctx)
    elif llm_fn is not None:
        ans = llm_fn(state["question"], summary_ctx, model_embedding)
    else:
        ans = ollama_llm(state["question"], context=summary_ctx, model_embedding=model_embedding)
    return {"generation": ans, "trace": state.get("trace", []) + ["codebase_ast"]}


def retrieve_node(
    state: AgentState,
    retriever: Optional[RetrieverProtocol] = None
) -> dict:
    q = state.get("search_query") or state["question"]
    docs = []
    if retriever is not None:
        try:
            if hasattr(retriever, "retrieve_and_rerank"):
                docs = retriever.retrieve_and_rerank(q, retrieve_limit=50, rerank_top_k=10)
                print(f"🔍 [CRAG Retrieval] Query='{q}' | Candidates retrieved & reranked: {len(docs)}")
                for idx, doc in enumerate(docs[:3]):
                    snippet = doc.get("text", "").replace("\n", " ")[:150]
                    print(f"   ├─ Doc {idx+1} [score={doc.get('score', 0):.4f}]: {snippet}...")
            elif hasattr(retriever, "invoke"):
                raw = retriever.invoke(q)
                docs = [{"id": i, "text": d.page_content, "score": 1.0} for i, d in enumerate(raw)]
        except Exception as e:
            print(f"⚠️ Vector retrieval failure: {e}")
            docs = [{"id": "fallback", "text": f"Retrieval error: {e}", "metadata": {"error": str(e)}}]
    return {"documents": docs, "trace": state.get("trace", []) + [f"retrieve:{len(docs)}"]}


def grade_node(
    state: AgentState,
    grader_fn: Optional[Callable] = None,
    model_embedding: str = "qwen2.5:7b-fenced"
) -> dict:
    docs = state.get("documents", [])
    _grader = grader_fn or default_grade_documents
    is_relevant, relevant_docs = _grader(state["question"], docs, model_name=model_embedding)
    return {
        "is_relevant": is_relevant,
        "documents": relevant_docs,
        "trace": state.get("trace", []) + [f"grade:{is_relevant}"]
    }


def rewrite_node(
    state: AgentState,
    rewriter_fn: Optional[Callable] = None,
    model_embedding: str = "qwen2.5:7b-fenced"
) -> dict:
    retries = state.get("retry_count", 0) + 1
    _rewriter = rewriter_fn or default_rewrite_query
    rewritten = _rewriter(state["question"], model_name=model_embedding)
    return {
        "search_query": rewritten,
        "retry_count": retries,
        "trace": state.get("trace", []) + [f"rewrite:{retries}"]
    }


def generate_node(
    state: AgentState,
    llm_provider: Optional[LLMProviderProtocol] = None,
    llm_fn: Optional[Callable] = None,
    model_embedding: str = "qwen2.5:7b-fenced"
) -> dict:
    docs = state.get("documents", [])
    formatted_content = "\n\n".join(d.get("text", "") for d in docs if d.get("text"))
    if llm_provider is not None:
        ans = llm_provider.generate(state["question"], context=formatted_content)
    elif llm_fn is not None:
        ans = llm_fn(state["question"], formatted_content, model_embedding)
    else:
        ans = ollama_llm(state["question"], formatted_content, model_embedding)
    return {"generation": ans, "trace": state.get("trace", []) + ["generate"]}


def fallback_node(state: AgentState) -> dict:
    msg = "I could not find sufficient relevant context in the provided documents to answer your question accurately."
    return {"generation": msg, "trace": state.get("trace", []) + ["fallback_exhausted"]}


def build_crag_graph(
    retriever: Optional[RetrieverProtocol] = None,
    llm_provider: Optional[LLMProviderProtocol] = None,
    model_embedding: str = "qwen2.5:7b-fenced",
    max_retries: int = 2,
    grader_fn: Optional[Callable] = None,
    rewriter_fn: Optional[Callable] = None,
    router_fn: Optional[Callable] = None,
    llm_fn: Optional[Callable] = None,
) -> Any:
    """
    Constructs the Corrective RAG (CRAG) state machine with injected dependencies.
    """
    if StateGraph is None:
        return None

    builder = StateGraph(AgentState)

    # SOLID: DIP — Inject abstractions via partial application
    builder.add_node(
        "route_node",
        partial(route_node, router_fn=router_fn, model_embedding=model_embedding)
    )
    builder.add_node(
        "direct_chat_node",
        partial(direct_chat_node, llm_provider=llm_provider, llm_fn=llm_fn, model_embedding=model_embedding)
    )
    builder.add_node(
        "codebase_ast_node",
        partial(codebase_ast_node, llm_provider=llm_provider, llm_fn=llm_fn, model_embedding=model_embedding)
    )
    builder.add_node("retrieve_node", partial(retrieve_node, retriever=retriever))
    builder.add_node(
        "grade_node",
        partial(grade_node, grader_fn=grader_fn, model_embedding=model_embedding)
    )
    builder.add_node(
        "rewrite_node",
        partial(rewrite_node, rewriter_fn=rewriter_fn, model_embedding=model_embedding)
    )
    builder.add_node(
        "generate_node",
        partial(generate_node, llm_provider=llm_provider, llm_fn=llm_fn, model_embedding=model_embedding)
    )
    builder.add_node("fallback_node", fallback_node)

    builder.set_entry_point("route_node")

    def route_decision(s: AgentState) -> str:
        r = s.get("route", "vector_search")
        if r == "general_chat":
            return "direct_chat_node"
        elif r == "codebase_ast":
            return "codebase_ast_node"
        return "retrieve_node"

    builder.add_conditional_edges("route_node", route_decision, {
        "direct_chat_node": "direct_chat_node",
        "codebase_ast_node": "codebase_ast_node",
        "retrieve_node": "retrieve_node"
    })

    builder.add_edge("direct_chat_node", END)
    builder.add_edge("codebase_ast_node", END)
    builder.add_edge("retrieve_node", "grade_node")

    def grade_decision(s: AgentState) -> str:
        if s.get("is_relevant"):
            return "generate_node"
        elif s.get("retry_count", 0) < s.get("max_retries", max_retries):
            return "rewrite_node"
        return "fallback_node"

    builder.add_conditional_edges("grade_node", grade_decision, {
        "generate_node": "generate_node",
        "rewrite_node": "rewrite_node",
        "fallback_node": "fallback_node"
    })

    builder.add_edge("rewrite_node", "retrieve_node")
    builder.add_edge("generate_node", END)
    builder.add_edge("fallback_node", END)

    return builder.compile()


def run_crag_agent(
    question: str,
    retriever: Optional[RetrieverProtocol] = None,
    llm_provider: Optional[LLMProviderProtocol] = None,
    model_embedding: str = "qwen2.5:7b-fenced",
    max_retries: int = 2,
    grader_fn: Optional[Callable] = None,
    rewriter_fn: Optional[Callable] = None,
    router_fn: Optional[Callable] = None,
    llm_fn: Optional[Callable] = None,
) -> Tuple[str, List[str]]:
    """
    Executes the self-correcting CRAG graph.
    Returns (generation_text, execution_trace).
    """
    _router = router_fn or default_route_query
    _grader = grader_fn or default_grade_documents
    _rewriter = rewriter_fn or default_rewrite_query

    graph = build_crag_graph(
        retriever=retriever,
        llm_provider=llm_provider,
        model_embedding=model_embedding,
        max_retries=max_retries,
        grader_fn=_grader,
        rewriter_fn=_rewriter,
        router_fn=_router,
        llm_fn=llm_fn,
    )

    if graph is not None:
        initial_state: AgentState = {
            "question": question,
            "search_query": question,
            "route": "vector_search",
            "documents": [],
            "is_relevant": False,
            "generation": "",
            "retry_count": 0,
            "max_retries": max_retries,
            "trace": []
        }
        final_state = graph.invoke(initial_state)
        return final_state.get("generation", ""), final_state.get("trace", [])

    # Deterministic fallback loop if StateGraph is unavailable
    route = _router(question, model_name=model_embedding)
    if route == "general_chat":
        if llm_provider is not None:
            return llm_provider.generate(question, ""), ["route:general_chat", "direct_chat"]
        if llm_fn is not None:
            return llm_fn(question, "", model_embedding), ["route:general_chat", "direct_chat"]
        return ollama_llm(question, "", model_embedding), ["route:general_chat", "direct_chat"]
    if route == "codebase_ast":
        ctx = "Codebase architecture map"
        if llm_provider is not None:
            return llm_provider.generate(question, ctx), ["route:codebase_ast", "codebase_ast"]
        if llm_fn is not None:
            return llm_fn(question, ctx, model_embedding), ["route:codebase_ast", "codebase_ast"]
        return ollama_llm(question, ctx, model_embedding), ["route:codebase_ast", "codebase_ast"]

    if retriever is None:
        return "Error: Retriever is not initialized.", ["route:vector_search", "no_retriever"]

    retries = 0
    search_q = question
    trace = [f"route:{route}"]

    while retries <= max_retries:
        if hasattr(retriever, "retrieve_and_rerank"):
            docs = retriever.retrieve_and_rerank(search_q, retrieve_limit=50, rerank_top_k=10)
        elif hasattr(retriever, "invoke"):
            docs = [{"id": i, "text": d.page_content, "score": 1.0} for i, d in enumerate(retriever.invoke(search_q))]
        else:
            return "Error: Unsupported retriever interface.", trace

        trace.append(f"retrieve:{len(docs)}")
        is_relevant, relevant_docs = _grader(question, docs, model_name=model_embedding)
        trace.append(f"grade:{is_relevant}")

        if is_relevant:
            formatted_content = "\n\n".join(d.get("text", "") for d in relevant_docs if d.get("text"))
            if llm_provider is not None:
                ans = llm_provider.generate(question, formatted_content)
            elif llm_fn is not None:
                ans = llm_fn(question, formatted_content, model_embedding)
            else:
                ans = ollama_llm(question, formatted_content, model_embedding)
            trace.append("generate")
            return ans, trace

        retries += 1
        if retries <= max_retries:
            search_q = _rewriter(question, model_name=model_embedding)
            trace.append(f"rewrite:{retries}")

    trace.append("fallback_exhausted")
    return "I could not find sufficient relevant context in the provided documents to answer your question accurately.", trace


def rag_chain(
    question: str,
    text_splitter: Any,
    retriever: Any,
    model_embedding: str = "qwen2.5:7b-fenced",
    llm_fn: Optional[Callable] = None,
) -> str:
    """
    CRAG pipeline entrypoint preserving polymorphic caller contracts.
    """
    generation, _ = run_crag_agent(
        question,
        retriever=retriever,
        model_embedding=model_embedding,
        max_retries=2,
        llm_fn=llm_fn,
    )
    return generation
