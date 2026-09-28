#!/usr/bin/env python3
"""
Pytest Suite & Chaos Simulation for Zero-VRAM Hybrid Retrieval and System Parsers.
Validates:
1. Qdrant 500 / timeout error resilience and idempotent fallback response.
2. Frontmatter parser identifying malformed vs valid YAML in memory.md.
3. CPU concurrency thread fencing (threads=4) and Zero-VRAM retrieval pipeline.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.context_assembler import (
    ZeroVRAMRetriever,
    parse_memory_frontmatter,
)


# ============================================================
# CHAOS SIMULATION: QDRANT 500 / TIMEOUT RESILIENCE
# ============================================================

def test_qdrant_500_timeout_fallback_sync():
    """
    Chaos Test: Verify that when Qdrant returns a 500 Server Error or Timeout,
    the retriever catches the exception and returns a structured fallback instead of crashing.
    """
    mock_client = MagicMock()
    mock_client.search.side_effect = TimeoutError("HTTP 500 Server Error: Qdrant vector store connection timed out")

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    retriever = ZeroVRAMRetriever(
        client=mock_client,
        embedding_model=mock_embedder,
        threads=4
    )

    # Retrieval must not raise unhandled exception
    results = retriever.retrieve_candidates("test query for chaos simulation")

    assert len(results) == 1
    assert results[0]["id"] == "fallback"
    assert "error" in results[0]["metadata"]
    assert "TimeoutError" in results[0]["metadata"]["error"]


@pytest.mark.asyncio
async def test_qdrant_async_chaos_and_reranker_fallback():
    """
    Async Chaos Test: Verify async workflow resilience under downstream Qdrant and FlashRank failures.
    """
    mock_client = MagicMock()
    mock_client.search.side_effect = ConnectionRefusedError("Connection refused by Qdrant at 6333")

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.05] * 384]

    mock_reranker = MagicMock()
    mock_reranker.rerank.side_effect = RuntimeError("CPU Reranker execution failure")

    retriever = ZeroVRAMRetriever(
        client=mock_client,
        embedding_model=mock_embedder,
        reranker=mock_reranker,
        threads=4
    )

    # Execute full retrieve and rerank pipeline
    final_docs = retriever.retrieve_and_rerank("What is the system architecture?", retrieve_limit=20, rerank_top_k=5)

    assert len(final_docs) == 1
    assert final_docs[0]["id"] == "fallback"
    assert final_docs[0]["metadata"].get("fallback") is True


# ============================================================
# MEMORY.MD FRONTMATTER PARSER TESTS
# ============================================================

def test_memory_frontmatter_valid_yaml():
    """
    Verify that well-formed YAML frontmatter in memory.md is correctly parsed.
    """
    valid_content = """---
model: "qwen2.5:7b-instruct-q4_K_M"
context_window: 4096
temperature: 0.2
roles:
  - generator
  - chaos_auditor
---
# Memory Body Header
System memory context persists across container boots.
"""
    result = parse_memory_frontmatter(valid_content)

    assert isinstance(result, dict)
    assert "metadata" in result
    assert "body" in result
    assert result["metadata"]["model"] == "qwen2.5:7b-instruct-q4_K_M"
    assert result["metadata"]["context_window"] == 4096
    assert result["metadata"]["temperature"] == 0.2
    assert "generator" in result["metadata"]["roles"]
    assert "# Memory Body Header" in result["body"]


def test_memory_frontmatter_malformed_yaml_syntax():
    """
    Verify that malformed YAML syntax in memory.md raises ValueError with details.
    """
    malformed_yaml_content = """---
model: "broken
  unclosed_quote: true
  invalid_tab:\t\t[
---
# Body
Some text.
"""
    with pytest.raises(ValueError, match="Malformed YAML frontmatter"):
        parse_memory_frontmatter(malformed_yaml_content)


def test_memory_frontmatter_unclosed_delimiter():
    """
    Verify that unclosed frontmatter delimiter '---' raises ValueError.
    """
    unclosed_content = """---
title: "Unclosed Header"
context: 4096
# Missing closing triple-dashes
"""
    with pytest.raises(ValueError, match="unclosed delimiter"):
        parse_memory_frontmatter(unclosed_content)


def test_memory_frontmatter_non_dict_metadata():
    """
    Verify that non-dict YAML (e.g. a plain list or scalar) raises ValueError.
    """
    non_dict_yaml = """---
- item 1
- item 2
---
Body text
"""
    with pytest.raises(ValueError, match="expected dict mapping"):
        parse_memory_frontmatter(non_dict_yaml)


# ============================================================
# THREAD FENCING & ZERO-VRAM PIPELINE EXECUTION
# ============================================================

def test_zero_vram_retriever_threads_and_pipeline():
    """
    Verify ZeroVRAMRetriever adheres to threads=4 limit and successfully
    processes retrieval and reranking with mocked Qdrant vector responses.
    """
    import os
    assert os.environ.get("OMP_NUM_THREADS") == "4"

    # Mock Qdrant search results
    class MockScoredPoint:
        def __init__(self, point_id, score, text):
            self.id = point_id
            self.score = score
            self.payload = {"text": text, "source": "test_doc.pdf"}

    mock_client = MagicMock()
    mock_client.search.return_value = [
        MockScoredPoint(i, 0.9 - (i * 0.02), f"Retrieved passage content #{i}")
        for i in range(20)
    ]

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1] * 384]

    mock_reranker = MagicMock()
    # FlashRank returns reranked list of dicts with 'id', 'text', 'score', 'meta'
    mock_reranker.rerank.return_value = [
        {"id": 3, "text": "Retrieved passage content #3", "score": 0.98, "meta": {"source": "test_doc.pdf"}},
        {"id": 0, "text": "Retrieved passage content #0", "score": 0.95, "meta": {"source": "test_doc.pdf"}},
        {"id": 7, "text": "Retrieved passage content #7", "score": 0.91, "meta": {"source": "test_doc.pdf"}},
        {"id": 1, "text": "Retrieved passage content #1", "score": 0.88, "meta": {"source": "test_doc.pdf"}},
        {"id": 5, "text": "Retrieved passage content #5", "score": 0.85, "meta": {"source": "test_doc.pdf"}},
    ]

    retriever = ZeroVRAMRetriever(
        client=mock_client,
        embedding_model=mock_embedder,
        reranker=mock_reranker,
        threads=4
    )

    top_5 = retriever.retrieve_and_rerank("Query testing zero VRAM", retrieve_limit=20, rerank_top_k=5)

    assert len(top_5) == 5
    assert top_5[0]["id"] == 3
    assert top_5[0]["score"] == 0.98
    assert mock_client.search.called
    assert mock_reranker.rerank.called


def test_rag_chain_with_zero_vram_retriever(monkeypatch):
    """
    Verify rag_chain in run_rag.py accepts ZeroVRAMRetriever and formats prompt context correctly.
    """
    from unittest.mock import MagicMock
    from run_rag import rag_chain

    mock_retriever = MagicMock()
    mock_retriever.retrieve_and_rerank.return_value = [
        {"id": 1, "text": "Passage 1 content", "score": 0.99},
        {"id": 2, "text": "Passage 2 content", "score": 0.95}
    ]

    mock_ollama_llm = MagicMock(return_value="Generated response from fenced model")
    monkeypatch.setattr("run_rag.ollama_llm", mock_ollama_llm)

    response = rag_chain("What is MLOps?", None, mock_retriever, "qwen2.5:7b-fenced")

    assert response == "Generated response from fenced model"
    mock_retriever.retrieve_and_rerank.assert_called_once_with("What is MLOps?", retrieve_limit=20, rerank_top_k=5)
    mock_ollama_llm.assert_called_once_with("What is MLOps?", "Passage 1 content\n\nPassage 2 content", "qwen2.5:7b-fenced")


def test_ollama_llm_runtime_options_injection(monkeypatch):
    """
    Verify that ollama_llm dynamically extracts OLLAMA_NUM_CTX and OLLAMA_TEMPERATURE
    from the environment and passes them to Client.chat(options=...).
    """
    import os
    from unittest.mock import MagicMock
    from run_rag import ollama_llm

    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "0.35")

    captured_kwargs = {}

    class MockClient:
        def __init__(self, host=None, timeout=None):
            pass

        def chat(self, **kwargs):
            nonlocal captured_kwargs
            captured_kwargs = kwargs
            return {"message": {"content": "Test response from 8k context model"}}

    monkeypatch.setattr("run_rag.Client", MockClient)

    res = ollama_llm("Explain neural networks", "Deep learning context", "qwen2.5:7b-fenced")

    assert res == "Test response from 8k context model"
    assert "options" in captured_kwargs
    assert captured_kwargs["options"]["num_ctx"] == 8192
    assert captured_kwargs["options"]["temperature"] == 0.35


def test_modelfile_template_rendering(tmp_path):
    """
    Verify that Modelfile.template produces a valid fenced Modelfile
    with environment variable interpolation for dynamic hardware scaling.
    """
    template_path = REPO_ROOT / "scripts" / "Modelfile.template"
    assert template_path.exists(), "scripts/Modelfile.template must exist"

    template_content = template_path.read_text(encoding="utf-8")

    # Simulate dynamic rendering as performed by entrypoint.sh
    num_ctx = "16384"
    temperature = "0.1"
    base_model = "qwen2.5:14b-instruct-q4_K_M"

    import re
    rendered = re.sub(r"\$\{OLLAMA_BASE_MODEL:-[^}]*\}", base_model, template_content)
    rendered = re.sub(r"\$\{OLLAMA_NUM_CTX:-[^}]*\}", num_ctx, rendered)
    rendered = re.sub(r"\$\{OLLAMA_TEMPERATURE:-[^}]*\}", temperature, rendered)

    assert f"FROM {base_model}" in rendered
    assert f"PARAMETER num_ctx {num_ctx}" in rendered
    assert f"PARAMETER temperature {temperature}" in rendered


def test_qdrant_universal_healthcheck_command():
    """
    Verify universal healthcheck command string conforms to POSIX syntax
    and contains all four fallback tiers: curl, wget, nc, and /dev/tcp.
    """
    import yaml
    compose_path = REPO_ROOT / "docker-compose.yaml"
    with open(compose_path, "r", encoding="utf-8") as f:
        compose = yaml.safe_load(f)

    healthcheck = compose["services"]["qdrant"]["healthcheck"]
    test_cmd = healthcheck["test"]

    assert test_cmd[0] == "CMD-SHELL"
    cmd_str = test_cmd[1]

    assert "curl" in cmd_str
    assert "wget" in cmd_str
    assert "nc" in cmd_str
    assert "/dev/tcp" in cmd_str
    assert "6333" in cmd_str


# ============================================================
# SRE & CHAOS TESTS: GRADIO CLEANUP & CACHE ALIGNMENT
# ============================================================

def test_gradio_temporary_file_active_cleanup(tmp_path, monkeypatch):
    """
    SRE Test: Verify cleanup_temporary_files deletes temporary upload artifacts
    and prunes empty parent directories inside tempdir, without touching external files.
    """
    import tempfile
    from run_rag import cleanup_temporary_files

    # 1. Create a simulated Gradio upload inside system tempdir
    temp_base = Path(tempfile.gettempdir())
    gradio_temp_dir = temp_base / "gradio" / "session_test_upload"
    gradio_temp_dir.mkdir(parents=True, exist_ok=True)
    temp_pdf = gradio_temp_dir / "uploaded_doc.pdf"
    temp_pdf.write_text("dummy pdf binary content", encoding="utf-8")

    assert temp_pdf.exists()

    # 2. Create external safe file (should NOT be unlinked)
    safe_file = tmp_path / "permanent_doc.pdf"
    safe_file.write_text("permanent content", encoding="utf-8")

    # Execute cleanup
    cleanup_temporary_files([str(temp_pdf), str(safe_file)])

    # Verify temp file and empty parent directory were purged
    assert not temp_pdf.exists(), "Temporary PDF must be deleted"
    assert not gradio_temp_dir.exists(), "Empty temporary upload directory must be pruned"

    # Verify safe file was untouched
    assert safe_file.exists(), "Safe external file must not be deleted"


def test_process_pdf_cleanup_on_loader_exception(monkeypatch):
    """
    Chaos Test: Verify that if PyPDFLoader raises an unhandled exception on a malformed PDF,
    the try...finally guard in process_pdf guarantees temporary files are still garbage-collected.
    """
    import tempfile
    from run_rag import process_pdf

    temp_base = Path(tempfile.gettempdir())
    leak_test_dir = temp_base / "gradio" / "corrupt_test_dir"
    leak_test_dir.mkdir(parents=True, exist_ok=True)
    corrupted_pdf = leak_test_dir / "corrupted.pdf"
    corrupted_pdf.write_text("%PDF-malformed-stream", encoding="utf-8")

    class MockFailingLoader:
        def __init__(self, file_path):
            self.file_path = file_path
        def load(self):
            raise RuntimeError("PDFSyntaxError: Malformed PDF stream corrupt")

    monkeypatch.setattr("run_rag.PyPDFLoader", MockFailingLoader)

    with pytest.raises(RuntimeError, match="PDFSyntaxError"):
        process_pdf(
            pdf_source=[str(corrupted_pdf)],
            model_embedding="qwen2.5:7b-fenced",
            persist_directory="/tmp/test_persist",
            chunk_size=500,
            chunk_overlap=100
        )

    # File must be deleted despite unhandled exception
    assert not corrupted_pdf.exists(), "Corrupted temporary file must be unlinked in finally block"
    assert not leak_test_dir.exists(), "Empty parent temp directory must be pruned in finally block"


def test_docker_compose_cache_alignment_config():
    """
    Verify docker-compose.yaml mounts FastEmbed cache using the configurable
    ${FASTEMBED_CACHE_DIR:-fastembed_cache}:/root/.cache/fastembed pattern
    across client and notebook services to enable bare metal host-to-container alignment.
    """
    compose_path = REPO_ROOT / "docker-compose.yaml"
    compose_text = compose_path.read_text(encoding="utf-8")

    expected_volume = "${FASTEMBED_CACHE_DIR:-fastembed_cache}:/root/.cache/fastembed"

    assert compose_text.count(expected_volume) == 2, (
        f"Expected {expected_volume} to appear in both client and notebook services in docker-compose.yaml"
    )



