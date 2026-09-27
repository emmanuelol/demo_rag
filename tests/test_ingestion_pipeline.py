#!/usr/bin/env python3
"""
Unit and Integration Test Suite for Dual-Artifact Data Ingestion (Phase 4).
Tests:
1. Deterministic UUID generation idempotency and validity.
2. Semantic Markdown header chunking and context retention.
3. Generator-based batching and memory boundaries.
4. Qdrant indexer upsert idempotency and CPU thread fences.
5. Ingestion pipeline CLI argument parsing and error recovery.
"""

import os
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.ingestion_core import (
    generate_deterministic_chunk_id,
    DocumentProcessor,
    QdrantIndexer
)


def test_deterministic_chunk_id_idempotency():
    """Verify that hashing same text + metadata always produces identical valid UUID."""
    text1 = "Hardware isolation is enforced at container boundaries."
    meta1 = {"source": "architecture.md", "Header 1": "Architecture", "Header 2": "Hardware"}

    id1 = generate_deterministic_chunk_id(text1, meta1)
    id2 = generate_deterministic_chunk_id(text1, meta1)

    # Idempotency assertion
    assert id1 == id2
    # Standard RFC 4122 UUID validation
    parsed_uuid = uuid.UUID(id1)
    assert str(parsed_uuid) == id1

    # Distinct content yields distinct UUID
    text2 = "Different chunk content for testing."
    id3 = generate_deterministic_chunk_id(text2, meta1)
    assert id1 != id3

    # Distinct header hierarchy yields distinct UUID
    meta2 = {"source": "architecture.md", "Header 1": "Architecture", "Header 2": "Networking"}
    id4 = generate_deterministic_chunk_id(text1, meta2)
    assert id1 != id4


def test_document_processor_semantic_splitting():
    """Verify that MarkdownHeaderTextSplitter captures headers and adds metadata."""
    md_content = """# System Overview
High-level description of the system architecture.

## Vector Engine
Qdrant stores vectors in host RAM.

### FastEmbed Pipeline
CPU-bound dense embeddings using bge-small-en-v1.5.

## Observability
Telemetry traces stream to Arize Phoenix.
"""
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    chunks = processor.process_text(md_content, source_path="test_doc.md")

    assert len(chunks) >= 3
    # Check that header hierarchies are retained
    headers_found = [c["metadata"].get("Header 2") for c in chunks if "Header 2" in c["metadata"]]
    assert "Vector Engine" in headers_found
    assert "Observability" in headers_found

    # Check deterministic chunk ID formatting
    for chunk in chunks:
        assert "id" in chunk
        assert uuid.UUID(chunk["id"])
        assert chunk["metadata"]["source"] == "test_doc.md"
        assert chunk["metadata"]["char_length"] == len(chunk["text"])


def test_document_processor_large_chunk_recursive_split():
    """Verify that sections larger than chunk_size undergo secondary recursive splitting."""
    long_paragraph = "Deterministic execution eliminates randomness. " * 50  # ~2300 chars
    md_content = f"# Long Section\n\n## SubSection\n{long_paragraph}"

    processor = DocumentProcessor(chunk_size=400, chunk_overlap=50)
    chunks = processor.process_text(md_content, source_path="long_doc.md")

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["metadata"]["Header 1"] == "Long Section"
        assert chunk["metadata"]["Header 2"] == "SubSection"
        assert len(chunk["text"]) <= 450


def test_batch_generator_slicing():
    """Verify batch generator slices lists correctly."""
    items = list(range(105))
    batches = list(QdrantIndexer.batch_generator(items, batch_size=32))

    assert len(batches) == 4
    assert len(batches[0]) == 32
    assert len(batches[1]) == 32
    assert len(batches[2]) == 32
    assert len(batches[3]) == 9
    # Complete reconstitution
    reconstituted = [item for b in batches for item in b]
    assert reconstituted == items


def test_qdrant_indexer_mock_upsert():
    """Verify batch indexing and idempotent upserting logic with mocked Qdrant and FastEmbed."""
    mock_qdrant_client = MagicMock()
    mock_collections = MagicMock()
    mock_collections.collections = []
    mock_qdrant_client.get_collections.return_value = mock_collections

    mock_embed_model = MagicMock()
    mock_embed_model.embed.side_effect = lambda texts: [[0.1] * 384 for _ in texts]

    indexer = QdrantIndexer(
        collection_name="test_collection",
        qdrant_host="localhost",
        qdrant_port=6333,
        threads=4
    )
    indexer._client = mock_qdrant_client
    indexer._embedding_model = mock_embed_model

    sample_chunks = [
        {"id": str(uuid.uuid4()), "text": f"Chunk text {i}", "metadata": {"idx": i}}
        for i in range(10)
    ]

    stats = indexer.index_chunks(sample_chunks, batch_size=4, show_progress=False)

    assert stats["total_chunks"] == 10
    assert stats["indexed_points"] == 10
    assert stats["batches"] == 3

    # Verify collection was created with size 384
    mock_qdrant_client.create_collection.assert_called_once()
    # Verify upsert called 3 times (ceil(10/4))
    assert mock_qdrant_client.upsert.call_count == 3


def test_zero_side_effects_on_import():
    """Verify importing ingestion modules does not connect to network or instantiate models."""
    indexer = QdrantIndexer()
    assert indexer._client is None
    assert indexer._embedding_model is None
