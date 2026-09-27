#!/usr/bin/env python3
"""
Production Data Ingestion Pipeline CLI.
Batch-processes markdown documentation and upserts dense embeddings into Qdrant.
Enforces deterministic IDs (idempotent upserts) and strict CPU thread fencing.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Adjust path for internal module imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.ingestion_core import DocumentProcessor, QdrantIndexer


def setup_logger(verbose: bool = False) -> logging.Logger:
    """Configures structured stdout logging."""
    logger = logging.getLogger("ingest_pipeline")
    log_level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(log_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)

    # Avoid duplicate handlers if reconfigured
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Production Data Ingestion Pipeline for Zero-VRAM Agentic RAG"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./docs",
        help="Path to input directory containing Markdown files (default: ./docs)"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="demo_collection",
        help="Target Qdrant collection name (default: demo_collection)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding generation and vector upserts (default: 64)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Max character size per chunk (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Character overlap between chunk splits (default: 100)"
    )
    parser.add_argument(
        "--qdrant-host",
        type=str,
        default=os.getenv("QDRANT_HOST", "localhost"),
        help="Qdrant host address (default: localhost or $QDRANT_HOST)"
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=int(os.getenv("QDRANT_PORT", "6333")),
        help="Qdrant HTTP port (default: 6333 or $QDRANT_PORT)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="CPU thread limit for FastEmbed execution (default: 4)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(verbose=args.verbose)

    input_path = Path(args.input_dir).resolve()
    logger.info("==================================================")
    logger.info("🚀 Initiating Dual-Artifact Data Ingestion Pipeline")
    logger.info("==================================================")
    logger.info(f"📁 Input Directory: {input_path}")
    logger.info(f"📦 Qdrant Target:   http://{args.qdrant_host}:{args.qdrant_port} -> [{args.collection_name}]")
    logger.info(f"⚙️  Batch Size:      {args.batch_size} (CPU Threads: {args.threads})")

    if not input_path.exists():
        logger.error(f"❌ Input directory does not exist: {input_path}")
        sys.exit(1)

    # Step 1: Semantic Markdown Chunking
    logger.info("🔍 Step 1: Scanning and chunking markdown documentation...")
    processor = DocumentProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    try:
        if input_path.is_file():
            chunks = processor.process_file(input_path)
        else:
            chunks = processor.process_directory(input_path, recursive=True)
    except Exception as e:
        logger.error(f"❌ Document processing failed: {e}")
        sys.exit(1)

    if not chunks:
        logger.warning(f"⚠️ No markdown chunks extracted from {input_path}. Please verify markdown files exist.")
        logger.info("🏁 Pipeline completed with 0 points ingested.")
        sys.exit(0)

    logger.info(f"✅ Extracted {len(chunks)} semantic chunks with deterministic IDs.")

    # Step 2: CPU-Fenced Vector Ingestion into Qdrant
    logger.info("🧠 Step 2: Initializing FastEmbed and batch upserting into Qdrant...")
    try:
        indexer = QdrantIndexer(
            collection_name=args.collection_name,
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            threads=args.threads
        )
        result = indexer.index_chunks(
            chunks=chunks,
            batch_size=args.batch_size,
            show_progress=True
        )
    except Exception as e:
        logger.error(f"❌ Qdrant vector indexing failed: {e}")
        sys.exit(1)

    logger.info("==================================================")
    logger.info("🎉 Ingestion Pipeline Completed Successfully!")
    logger.info(f"   Total Chunks:   {result.get('total_chunks', 0)}")
    logger.info(f"   Indexed Points: {result.get('indexed_points', 0)}")
    logger.info(f"   Total Batches:  {result.get('batches', 0)}")
    logger.info(f"   Target Coll:    {result.get('collection')}")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
