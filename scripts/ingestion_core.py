#!/usr/bin/env python3
"""
Dual-Artifact Data Ingestion Core - Decoupled ETL Engine for RAG.
Provides:
1. Markdown-aware semantic chunking with header hierarchy preservation.
2. Deterministic MD5-based UUID generation for idempotent Qdrant ingestion.
3. Generator-based batch vectorization using CPU-fenced FastEmbed (threads=4).
4. Safe idempotent upserts into Qdrant without side-effects on import.
"""

import os
import uuid
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional, Union

# Hardware isolation: CPU thread fences to prevent thread starvation
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

logger = logging.getLogger("ingestion_core")


def generate_deterministic_chunk_id(chunk_text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Generates a deterministic RFC 4122 v5/MD5 UUID based on chunk content and metadata.
    Guarantees idempotency: exact same chunk and origin yields identical UUID.
    """
    meta = metadata or {}
    source = str(meta.get("source", "")).strip()
    h1 = str(meta.get("Header 1", "")).strip()
    h2 = str(meta.get("Header 2", "")).strip()
    h3 = str(meta.get("Header 3", "")).strip()

    # Composite fingerprint
    raw_key = f"{source}::{h1}::{h2}::{h3}::{chunk_text.strip()}"
    digest = hashlib.md5(raw_key.encode("utf-8")).hexdigest()
    return str(uuid.UUID(hex=digest))


class DocumentProcessor:
    """
    Extracts and chunks Markdown documentation while maintaining semantic header context.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        headers_to_split_on: Optional[List[tuple]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.headers_to_split_on = headers_to_split_on or [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]

    def _get_splitters(self):
        try:
            from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
        except ImportError:
            try:
                from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
            except ImportError as e:
                raise ImportError(
                    "langchain-text-splitters is required for DocumentProcessor. "
                    "Install with: pip install langchain-text-splitters"
                ) from e

        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False
        )
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        return header_splitter, recursive_splitter

    def process_text(self, markdown_text: str, source_path: str = "memory") -> List[Dict[str, Any]]:
        """
        Processes raw markdown string into structured semantic chunks with deterministic IDs.
        """
        if not markdown_text or not markdown_text.strip():
            return []

        header_splitter, recursive_splitter = self._get_splitters()
        header_docs = header_splitter.split_text(markdown_text)

        chunks: List[Dict[str, Any]] = []
        chunk_index = 0

        for doc in header_docs:
            content = getattr(doc, "page_content", str(doc))
            base_meta = dict(getattr(doc, "metadata", {}))
            base_meta["source"] = source_path

            if len(content) > self.chunk_size:
                sub_docs = recursive_splitter.split_text(content)
                for sub_text in sub_docs:
                    sub_meta = dict(base_meta)
                    sub_meta["chunk_index"] = chunk_index
                    sub_meta["char_length"] = len(sub_text)
                    chunk_id = generate_deterministic_chunk_id(sub_text, sub_meta)
                    chunks.append({
                        "id": chunk_id,
                        "text": sub_text,
                        "metadata": sub_meta
                    })
                    chunk_index += 1
            else:
                base_meta["chunk_index"] = chunk_index
                base_meta["char_length"] = len(content)
                chunk_id = generate_deterministic_chunk_id(content, base_meta)
                chunks.append({
                    "id": chunk_id,
                    "text": content,
                    "metadata": base_meta
                })
                chunk_index += 1

        return chunks

    def process_file(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Reads and chunks a single markdown file."""
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Markdown file does not exist: {path}")

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error(f"Failed to read file {path}: {e}")
            raise

        return self.process_text(content, source_path=path.name)

    def process_directory(self, directory_path: Union[str, Path], recursive: bool = True) -> List[Dict[str, Any]]:
        """Traverses a directory to chunk all markdown files."""
        dir_path = Path(directory_path).resolve()
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Directory does not exist: {dir_path}")

        pattern = "**/*.md" if recursive else "*.md"
        all_files = sorted(dir_path.glob(pattern))

        results: List[Dict[str, Any]] = []
        for file in all_files:
            file_chunks = self.process_file(file)
            results.extend(file_chunks)

        logger.info(f"Processed {len(all_files)} files into {len(results)} chunks from {dir_path}")
        return results


class QdrantIndexer:
    """
    CPU-fenced indexer connecting to local Qdrant and generating batch embeddings via FastEmbed.
    """

    def __init__(
        self,
        collection_name: str = "demo_collection",
        qdrant_host: Optional[str] = None,
        qdrant_port: Optional[int] = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        threads: int = 4
    ):
        self.collection_name = collection_name
        self.qdrant_host = qdrant_host or os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(qdrant_port or os.getenv("QDRANT_PORT", "6333"))
        self.embedding_model_name = embedding_model
        self.threads = threads

        self._client = None
        self._embedding_model = None

    @property
    def client(self):
        """Lazy-loaded Qdrant client."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                self._client = QdrantClient(
                    host=self.qdrant_host,
                    port=self.qdrant_port,
                    timeout=10.0,
                    check_compatibility=False
                )
            except Exception as e:
                logger.error(f"Failed to initialize QdrantClient({self.qdrant_host}:{self.qdrant_port}): {e}")
                raise
        return self._client

    @property
    def embedding_model(self):
        """Lazy-loaded FastEmbed model with strict thread limits."""
        if self._embedding_model is None:
            try:
                from fastembed import TextEmbedding
                self._embedding_model = TextEmbedding(
                    model_name=self.embedding_model_name,
                    threads=self.threads
                )
            except Exception as e:
                logger.error(f"Failed to initialize FastEmbed({self.embedding_model_name}): {e}")
                raise
        return self._embedding_model

    def ensure_collection(self, vector_size: int = 384) -> bool:
        """Verifies collection exists; creates it with Cosine distance if absent."""
        try:
            from qdrant_client.http import models as qmodels
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                logger.info(f"Creating Qdrant collection '{self.collection_name}' (dim={vector_size}, distance=Cosine)...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=vector_size,
                        distance=qmodels.Distance.COSINE
                    )
                )
            return True
        except Exception as e:
            logger.error(f"Failed to ensure collection '{self.collection_name}': {e}")
            raise

    @staticmethod
    def batch_generator(items: List[Any], batch_size: int = 64) -> Generator[List[Any], None, None]:
        """Yields chunks of items in slices of batch_size."""
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    def index_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 64,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Embeds and upserts chunks in batches with deterministic UUIDs into Qdrant.
        """
        if not chunks:
            logger.warning("Empty chunk list provided to index_chunks. Skipping.")
            return {"total_chunks": 0, "indexed_points": 0, "batches": 0}

        self.ensure_collection(vector_size=384)

        try:
            from qdrant_client.http import models as qmodels
        except ImportError as e:
            raise ImportError("qdrant-client is required for index_chunks") from e

        total_chunks = len(chunks)
        batches = list(self.batch_generator(chunks, batch_size=batch_size))
        total_batches = len(batches)

        progress_iter = batches
        if show_progress:
            try:
                from tqdm import tqdm
                progress_iter = tqdm(batches, desc=f"Upserting {self.collection_name}", unit="batch")
            except ImportError:
                pass

        indexed_count = 0
        for batch in progress_iter:
            texts = [c["text"] for c in batch]
            # FastEmbed generator to list
            embeddings = list(self.embedding_model.embed(texts))

            points = []
            for chunk, emb in zip(batch, embeddings):
                vector = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                payload = {
                    "text": chunk["text"],
                    "id": chunk["id"],
                    **chunk.get("metadata", {})
                }
                points.append(
                    qmodels.PointStruct(
                        id=chunk["id"],
                        vector=vector,
                        payload=payload
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            indexed_count += len(points)

        logger.info(f"Successfully indexed {indexed_count} points in {total_batches} batches.")
        return {
            "total_chunks": total_chunks,
            "indexed_points": indexed_count,
            "batches": total_batches,
            "collection": self.collection_name
        }
