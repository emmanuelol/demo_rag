# ADR-001: SOLID Architecture — LangGraph + Factory Pattern

**Date:** 2025-02  
**Status:** Accepted  
**Author:** Emmanuel  
**Context:** Industrial-Grade Corrective RAG (CRAG) System on Consumer Hardware

---

## Context

The initial prototype was a 1,048-line monolithic `run_rag.py` acting simultaneously as:
- A LangGraph state machine definition
- A Gradio UI event controller  
- A PDF ingestion pipeline
- A garbage collection handler for temp files

This god-object violated SRP, made unit testing impossible (every test required Gradio + Ollama + Qdrant running simultaneously), and caused a silent copy-paste bug where `_is_gradio_artifact()` was defined twice in the same file.

---

## Decision

Decompose `run_rag.py` into four strict layers using SOLID principles as the architectural guide:

### 1. Composition Root — `run_rag.py`
The entry point is now under 80 lines. It:
- Parses CLI args
- Instantiates the factory
- Injects LLM and retriever into the graph
- Launches the UI

It owns **zero business logic**. It is a wiring harness.

### 2. CRAG Graph — `scripts/crag_graph.py`
Contains `AgentState`, `build_crag_graph()`, and `run_crag_agent()`. Graph nodes receive their dependencies (LLM, retriever) via `functools.partial` at build time — they never import concrete providers.

**Why `functools.partial` instead of closures?**  
Closures capture the variable by reference. If `llm_provider` is reassigned (e.g., during GCP fallback), closures silently pick up the new value — a source of subtle bugs in production. `functools.partial` captures the value at bind time, making injection deterministic and testable.

### 3. Protocols — `scripts/protocols.py`
Two `typing.Protocol` interfaces with `@runtime_checkable`:

```python
class RetrieverProtocol(Protocol):
    def retrieve_and_rerank(self, query: str, ...) -> List[Dict[str, Any]]: ...

class LLMProviderProtocol(Protocol):
    provider_type: str
    def generate(self, prompt: str, context: str = "") -> str: ...
```

**Why `Protocol` over `ABC`?**  
`abc.ABC` requires explicit inheritance (`class ZeroVRAMRetriever(BaseRetriever)`), which creates tight coupling to this codebase's base class. `typing.Protocol` provides structural subtyping — any class that implements the required methods conforms automatically. This allows third-party retrievers (e.g., LangChain's `BaseRetriever`) to satisfy the interface without modification.

### 4. UI Layer — `ui/gradio_app.py`
Contains `create_ui()` and all Gradio `gr.Blocks` event wiring. It imports `process_query` and `process_ingestion` from the pipeline layer but has zero knowledge of LangGraph internals.

### 5. Shared Utilities — `utils/`
- `utils/ui_helpers.py`: `_is_gradio_artifact()` and `cleanup_temporary_files()` — the duplicated bug is eliminated here
- `utils/config_loader.py`: `load_config()` with environment-segregated validation

---

## Consequences

### Positive

**Testability:** The CRAG graph can now be unit tested with zero running services:
```python
mock_llm = MagicMock(spec=LLMProviderProtocol)
mock_retriever = MagicMock(spec=RetrieverProtocol)
graph = build_crag_graph(llm_provider=mock_llm, retriever=mock_retriever)
```

**Headless CI/CD:** The LangGraph pipeline runs in Docker without instantiating Gradio. This reduced the GitHub Actions test matrix from requiring a full multi-container compose stack to a single container.

**Provider Swapping:** Switching from Local Ollama (RTX 4060) to GCP Vertex AI requires changing one environment variable (`DEPLOYMENT_ENV=gcp`). Zero lines of `crag_graph.py` change.

**Bug Elimination:** The `_is_gradio_artifact()` duplication (lines 188 and 669 of the original `run_rag.py`) is resolved by canonical definition in `utils/ui_helpers.py`.

### Negative / Trade-offs

**Import complexity increases:** Callers now import from four modules instead of one. This is an acceptable trade-off — the explicit import graph is the documentation.

**`functools.partial` is less discoverable:** New contributors may not immediately understand why nodes are wrapped with `partial`. A `# SOLID: DIP — injected at build time` comment on each `add_node` call mitigates this.

---

## Alternatives Considered

### LangChain `SequentialChain` instead of LangGraph `StateGraph`

Rejected. `SequentialChain` is a linear pipeline — it cannot model cyclic self-correction (retrieve → grade → rewrite → retrieve). LangGraph's directed cyclic graph is the correct primitive for CRAG. LangChain's own documentation now recommends LangGraph for agentic workflows.

### Single `BaseRetriever` ABC instead of Protocol

Rejected. Forces inheritance coupling on `ZeroVRAMRetriever`. `Protocol` achieves the same LSP guarantee without the coupling.

### Pydantic v2 `BaseModel` for config segregation

Deferred. Current `yaml` + env var approach is sufficient for a portfolio project. If this moves to production, `pydantic-settings` with separate `LocalSettings` and `GCPSettings` models is the correct upgrade path (see `.env.local.example` and `.env.gcp.example`).

---

## References

- [LangGraph — Cyclic Stateful Agents](https://langchain-ai.github.io/langgraph/)
- [Python `typing.Protocol` — PEP 544](https://peps.python.org/pep-0544/)
- [CRAG Paper — Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)
- `scripts/protocols.py` — Protocol definitions
- `scripts/crag_graph.py` — Extracted graph with DIP injection
- `scripts/factory.py` — Provider factory (OCP)
