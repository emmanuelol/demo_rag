# Zero-VRAM Agentic RAG System Architecture Guide

## Executive Overview
The Zero-VRAM Agentic RAG system is engineered for predictable, deterministic execution across consumer hardware and multi-cloud environments. By partitioning tasks across asymmetrical hardware resources, the platform eliminates CUDA Out-Of-Memory exceptions and guarantees high throughput.

## Hardware Isolation Boundaries
Hardware resource isolation is enforced at container boundaries:
- **RTX 4060 (8GB VRAM)**: Reserved exclusively for the generator LLM (fenced Qwen 2.5:7b). The context window is constrained via `num_ctx 4096` to prevent memory thrashing.
- **AMD Ryzen 7 (Host RAM)**: Dense vector generation via FastEmbed and cross-encoder reranking via FlashRank run exclusively on CPU threads.
- **Qdrant Vector Database**: Memory capped at 4GB in host RAM without exposing the NVIDIA CUDA container runtime.

## Dual-Artifact Ingestion Pipeline
The ingestion engine decouples semantic parsing from vector storage:
- **Semantic Markdown Chunking**: Preserves Markdown header hierarchy (`#`, `##`, `###`) to maintain contextual scope.
- **Deterministic Hashing**: Chunks generate RFC 4122 v5 UUIDs derived from MD5 hashes of chunk content and section hierarchy. Re-running the pipeline updates existing vectors idempotently without database duplication.
- **Batch Processing**: Streams chunks in batches of 64 or 128 through FastEmbed to maintain minimal memory overhead.

## Corrective RAG (CRAG) Workflow
Retrieved candidates undergo a rigorous self-correcting cycle:
1. **Semantic Routing**: Distinguishes between vector retrieval, direct chat, and AST inspection.
2. **Relevance Grading**: Evaluates retrieved document chunks against the prompt.
3. **Query Rewriting**: Reformulates ambiguous queries when candidate documents fail validation.
4. **Fallback Handling**: Gracefully informs the caller when context is insufficient to prevent hallucination.

## Telemetry & Observability
Non-blocking OpenTelemetry spans stream execution traces to an Arize Phoenix container on port 6006. In-memory buffers ensure that telemetry collection never introduces latency or halts the inference loop.
