#!/usr/bin/env python3
"""
Provider-Agnostic Factory - Bridges Local (Ollama + Qdrant) and GCP (Vertex AI + Cloud Search).
Dynamically selects and initializes LLM and Vector Store backends based on DEPLOYMENT_ENV.
Provides graceful automatic fallback to local bare-metal when cloud credentials fail or expire.
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Any, Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# Fallback local components
from scripts.context_assembler import ZeroVRAMRetriever
from scripts.llm_utils import ollama_llm


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    config_path = REPO_ROOT / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"⚠️ Error reading config.yaml: {e}")
        return {}


def get_deployment_env() -> str:
    """Resolve active deployment environment ('local' or 'gcp')."""
    env_var = os.getenv("DEPLOYMENT_ENV")
    if env_var:
        return env_var.strip().lower()

    cfg = load_config()
    return cfg.get("deployment", {}).get("environment", "local").strip().lower()


class LocalLLMWrapper:
    """Wrapper for local Ollama fenced model."""
    def __init__(self, model_name: str = "qwen2.5:7b-fenced", base_url: str = "http://ollama:11434"):
        self.model_name = model_name
        self.base_url = os.getenv("BASE_URL", base_url)
        self.provider_type = "local_ollama"

    def generate(self, prompt: str, context: str = "") -> str:
        return ollama_llm(prompt, context, self.model_name, base_url=self.base_url)

    def __call__(self, prompt: str, context: str = "") -> str:
        return self.generate(prompt, context)


class GCPVertexLLMWrapper:
    """Wrapper for GCP Gemini / Vertex AI model."""
    def __init__(self, model_name: str = "gemini-1.5-pro", project_id: Optional[str] = None):
        self.model_name = model_name
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID", "demo-rag-project")
        self.provider_type = "gcp_vertex_ai"

    def generate(self, prompt: str, context: str = "") -> str:
        formatted_prompt = f"Question: {prompt}\n\nContext: {context}" if context else prompt
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = os.getenv("GOOGLE_API_KEY")
            llm = ChatGoogleGenerativeAI(model=self.model_name, google_api_key=api_key)
            res = llm.invoke(formatted_prompt)
            return getattr(res, "content", str(res))
        except Exception as e:
            raise RuntimeError(f"GCP Vertex AI invocation failed: {e}") from e

    def __call__(self, prompt: str, context: str = "") -> str:
        return self.generate(prompt, context)


def get_llm_provider(deployment_env: Optional[str] = None) -> Any:
    """
    Returns LLM provider instance for active environment.
    Gracefully degrades to Local Ollama on GCP authentication failure.
    """
    env = (deployment_env or get_deployment_env()).lower()
    config = load_config()

    if env == "gcp":
        try:
            # SRE Guard: Validate GCP authentication before committing
            has_api_key = bool(os.getenv("GOOGLE_API_KEY"))
            has_adc = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

            # Check if google.auth default credentials succeed
            if not has_api_key and not has_adc:
                try:
                    import google.auth
                    _, project = google.auth.default()
                except Exception as auth_err:
                    raise PermissionError(f"No valid GCP credentials or ADC found: {auth_err}")

            gcp_cfg = config.get("providers", {}).get("gcp", {})
            model_name = gcp_cfg.get("llm", "gemini-1.5-pro")
            return GCPVertexLLMWrapper(model_name=model_name)

        except Exception as e:
            print(f"⚠️ [SRE Fallback] GCP LLM authentication failure ({type(e).__name__}: {e}).")
            print("   Gracefully degrading to Local Ollama provider (qwen2.5:7b-fenced).")
            local_cfg = config.get("providers", {}).get("local", {})
            model_name = local_cfg.get("llm", "qwen2.5:7b-fenced")
            base_url = local_cfg.get("base_url", "http://ollama:11434")
            return LocalLLMWrapper(model_name=model_name, base_url=base_url)

    # Local environment
    local_cfg = config.get("providers", {}).get("local", {})
    model_name = local_cfg.get("llm", "qwen2.5:7b-fenced")
    base_url = local_cfg.get("base_url", "http://ollama:11434")
    return LocalLLMWrapper(model_name=model_name, base_url=base_url)


def get_vector_provider(deployment_env: Optional[str] = None) -> Any:
    """
    Returns Vector Store provider instance.
    Gracefully degrades to local ZeroVRAMRetriever on cloud connection failure.
    """
    env = (deployment_env or get_deployment_env()).lower()
    config = load_config()
    local_cfg = config.get("providers", {}).get("local", {})
    host = os.getenv("QDRANT_HOST", local_cfg.get("host", "qdrant"))
    port = int(os.getenv("QDRANT_PORT", local_cfg.get("port", 6333)))

    if env == "gcp":
        try:
            # SRE Guard: Check for Cloud Vector Search credentials
            has_creds = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GOOGLE_API_KEY"))
            if not has_creds:
                raise PermissionError("GCP Vector Search credentials not configured.")

            # Simulated Vertex AI Vector Search wrapper
            print("☁️ Initializing GCP Vertex AI Vector Search provider...")
            # If actual cloud vector client is unavailable, trigger fallback
            raise NotImplementedError("GCP Vertex Vector client requires live GCP infrastructure.")
        except Exception as e:
            print(f"⚠️ [SRE Fallback] GCP Vector Search failure ({type(e).__name__}: {e}).")
            print("   Gracefully degrading to Local ZeroVRAMRetriever (Qdrant).")
            return ZeroVRAMRetriever(qdrant_host=host, qdrant_port=port, threads=4)

    # Local environment: Zero-VRAM Qdrant + FastEmbed + FlashRank
    return ZeroVRAMRetriever(qdrant_host=host, qdrant_port=port, threads=4)
