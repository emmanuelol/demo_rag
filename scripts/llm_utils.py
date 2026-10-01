#!/usr/bin/env python3
"""
LLM Utilities - Decoupled invocation and dispatch helpers for Ollama and Multi-Cloud LLMs.
Eliminates circular dependencies between run_rag.py and scripts/factory.py.
"""

import os
import re
from typing import Optional, Any

try:
    import ollama
    from ollama import Client, chat, ChatResponse
except ImportError:
    ollama = None
    Client = None
    chat = None
    ChatResponse = None


def resolve_ollama_base_url(custom_url: Optional[str] = None) -> str:
    """Resolves accessible Ollama endpoint across host, container, and local setups."""
    if custom_url:
        return custom_url
    env_url = os.getenv("OLLAMA_HOST") or os.getenv("BASE_URL")
    if env_url:
        # Normalize protocol if host:port passed
        if not env_url.startswith("http://") and not env_url.startswith("https://"):
            return f"http://{env_url}"
        return env_url
    return "http://localhost:11434"


def ollama_llm(
    question: str,
    context: str = "",
    model_embedding: str = "qwen2.5:7b-fenced",
    base_url: Optional[str] = None
) -> str:
    """
    Executes a direct prompt generation call against the Ollama daemon.
    Guarantees defensive error handling, reasoning trace filtering, and runtime parameter enforcement.
    """
    formatted_prompt = f"Question: {question}\n\nContext: {context}" if context else question
    resolved_url = resolve_ollama_base_url(base_url)

    if Client is None and chat is None:
        return f"Error: Ollama library not installed. Simulated answer for: {question}"

    try:
        # Parameterized runtime options to prevent VRAM over-allocation
        num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
        temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))

        client_kwargs: dict = {}
        if Client is not None:
            client_kwargs["client"] = Client(host=resolved_url, timeout=30.0)

        response: Any = chat(
            model=model_embedding,
            messages=[{"role": "user", "content": formatted_prompt}],
            options={"num_ctx": num_ctx, "temperature": temperature},
            **client_kwargs
        )

        response_content = (
            response.get("message", {}).get("content", "")
            if isinstance(response, dict)
            else getattr(getattr(response, "message", None), "content", "")
        )
        if not response_content:
            return "Error: Empty response received from LLM."

        # Strip reasoning traces if model produces them (e.g. DeepSeek R1)
        final_answer = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()
        return final_answer if final_answer else response_content.strip()
    except Exception as e:
        return f"Error: LLM service unreachable at {resolved_url} ({type(e).__name__}: {str(e)})"


def dispatch_llm_generation(
    prompt: str,
    context: str = "",
    model_embedding: str = "qwen2.5:7b-fenced"
) -> str:
    """
    Executes generation through the provider-agnostic factory.
    Gracefully falls back to direct ollama_llm on any initialization or invocation exception.
    """
    try:
        from scripts.factory import get_llm_provider
        provider = get_llm_provider()
        return provider.generate(prompt, context)
    except Exception:
        return ollama_llm(prompt, context, model_embedding)
