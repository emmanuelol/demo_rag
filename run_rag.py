# Libraries
import argparse
import os
import re
import tempfile
from pathlib import Path
from time import sleep
import yaml

from typing import TypedDict, List, Dict, Any, Optional, Tuple

try:
    import gradio as gr
except ImportError:
    gr = None

try:
    from langchain_chroma import Chroma
except ImportError:
    Chroma = None

try:
    from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
except ImportError:
    PyPDFDirectoryLoader = None
    PyPDFLoader = None

try:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
except ImportError:
    ChatOllama = None
    OllamaEmbeddings = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None

try:
    import ollama
    from ollama import ChatResponse, Client, chat
except ImportError:
    ollama = None
    Client = None
    chat = None
    ChatResponse = None

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    StateGraph = None
    END = None

from scripts.router import route_query
from scripts.grader import grade_documents, rewrite_query


def setup_telemetry(endpoint: Optional[str] = None) -> bool:
    """
    Initializes non-blocking OpenTelemetry instrumentation for LangChain/LangGraph with Phoenix.
    Enforces a strict 500ms timeout on the OTLP exporter to prevent blocking the event loop.
    """
    endpoint = endpoint or os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:6006")
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(
            endpoint=f"{endpoint.rstrip('/')}/v1/traces",
            timeout=float(os.getenv("OTEL_EXPORTER_OTLP_TIMEOUT", "500")) / 1000.0  # 500ms
        )
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        LangChainInstrumentor().instrument()
        print(f"📡 Telemetry initialized with non-blocking exporter targeting {endpoint} (timeout: 500ms)")
        return True
    except Exception as e:
        # Chaos guard: Never block execution if telemetry collector is unavailable
        return False


# Attempt telemetry initialization at module load
setup_telemetry()





def ollama_llm(question, context, model_embedding):
    formatted_prompt = f"Question: {question}\n\nContext: {context}" if context else question
    base_url = os.getenv('BASE_URL', 'http://ollama:11434')
    timeout = float(os.getenv('OLLAMA_TIMEOUT', '120.0'))
    num_ctx = int(os.getenv('OLLAMA_NUM_CTX', '4096'))
    temperature = float(os.getenv('OLLAMA_TEMPERATURE', '0.2'))

    try:
        client_cls = globals().get("Client")
        if client_cls is not None:
            client = client_cls(host=base_url, timeout=timeout)
            response = client.chat(
                model=model_embedding,
                messages=[{"role": "user", "content": formatted_prompt}],
                options={
                    "num_ctx": num_ctx,
                    "temperature": temperature,
                },
            )

            response_content = (
                response.get("message", {}).get("content", "")
                if isinstance(response, dict)
                else getattr(getattr(response, "message", None), "content", "")
            )
            if not response_content:
                return "Error: Empty response received from LLM."

            final_answer = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()
            return final_answer if final_answer else response_content.strip()
        else:
            return f"Error: Ollama library not installed. Simulated answer for: {question}"
    except Exception as e:
        return f"Error: LLM service unreachable or request failed ({type(e).__name__}: {str(e)})"


def dispatch_llm_generation(prompt: str, context: str = "", model_embedding: str = "qwen2.5:7b-fenced") -> str:
    """
    Executes generation through the provider-agnostic factory.
    Gracefully falls back to ollama_llm on any initialization or invocation exception.
    """
    try:
        from scripts.factory import get_llm_provider
        provider = get_llm_provider()
        return provider.generate(prompt, context)
    except Exception:
        return ollama_llm(prompt, context, model_embedding)


from utils.ui_helpers import is_gradio_artifact, cleanup_temporary_files


def process_pdf(pdf_source, model_embedding, persist_directory, chunk_size, chunk_overlap, reset_existing=False):
    """
    Extracts text from PDF source, chunks with text splitter, and indexes into Qdrant via Zero-VRAM FastEmbed.
    Automatically garbage-collects temporary uploaded PDF artifacts to prevent disk exhaustion.
    """
    if not pdf_source:
        return None, None, None

    data = []
    temp_files_to_clean = []

    try:
        if isinstance(pdf_source, list):
            for item in pdf_source:
                file_path = item.name if hasattr(item, 'name') else str(item)
                if os.path.exists(file_path):
                    if is_gradio_artifact(file_path):
                        temp_files_to_clean.append(os.path.abspath(file_path))

                    # Pre-flight check: skip 0-byte files to prevent EmptyFileError
                    try:
                        if os.path.getsize(file_path) == 0:
                            print(f"⚠️ Pre-flight check: Skipping 0-byte file: {file_path}")
                            continue
                    except OSError as e:
                        print(f"⚠️ Pre-flight error checking file size {file_path}: {e}")
                        continue

                    # Fault-tolerant document loader
                    if PyPDFLoader is not None:
                        try:
                            loader = PyPDFLoader(file_path=file_path)
                            loaded_docs = loader.load()
                            if loaded_docs:
                                data.extend(loaded_docs)
                        except Exception as e:
                            print(f"⚠️ Document loader error on {file_path} ({type(e).__name__}: {e}). Skipping file.")
        elif isinstance(pdf_source, str):
            if os.path.isdir(pdf_source):
                for p in sorted(Path(pdf_source).glob("**/*.pdf")):
                    f_str = str(p)
                    try:
                        if os.path.getsize(f_str) == 0:
                            print(f"⚠️ Pre-flight check: Skipping 0-byte file in directory: {f_str}")
                            continue
                        if PyPDFLoader is not None:
                            try:
                                loader = PyPDFLoader(file_path=f_str)
                                loaded_docs = loader.load()
                                if loaded_docs:
                                    data.extend(loaded_docs)
                            except Exception as e:
                                print(f"⚠️ Document loader error on {f_str} ({type(e).__name__}: {e}). Skipping file.")
                    except OSError as e:
                        print(f"⚠️ Pre-flight error checking file size {f_str}: {e}")
                        continue
            elif os.path.isfile(pdf_source):
                if is_gradio_artifact(pdf_source):
                    temp_files_to_clean.append(os.path.abspath(pdf_source))
                try:
                    if os.path.getsize(pdf_source) == 0:
                        print(f"⚠️ Pre-flight check: Skipping 0-byte file: {pdf_source}")
                    elif PyPDFLoader is not None:
                        try:
                            loader = PyPDFLoader(file_path=pdf_source)
                            loaded_docs = loader.load()
                            if loaded_docs:
                                data.extend(loaded_docs)
                        except Exception as e:
                            print(f"⚠️ Document loader error on {pdf_source} ({type(e).__name__}: {e}). Skipping file.")
                except OSError as e:
                    print(f"⚠️ Pre-flight error checking file size {pdf_source}: {e}")
            else:
                raise FileNotFoundError(f"Path not found: {pdf_source}")
        else:
            raise TypeError(f"Unsupported pdf_source type: {type(pdf_source)}")
    finally:
        # Chaos/SRE Guard: Unconditionally garbage-collect uploaded temp files even on load failure
        cleanup_temporary_files(temp_files_to_clean)

    if not data:
        return None, None, None

    text_splitter = None
    if RecursiveCharacterTextSplitter is not None:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(data)
    else:
        chunks = data

    from scripts.ingestion_core import QdrantIndexer, generate_deterministic_chunk_id
    from scripts.context_assembler import ZeroVRAMRetriever

    q_host = os.getenv("QDRANT_HOST", "qdrant" if os.path.exists("/.dockerenv") else "localhost")
    q_port = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name = os.getenv("QDRANT_COLLECTION", "demo_collection")

    indexer = QdrantIndexer(
        collection_name=collection_name,
        qdrant_host=q_host,
        qdrant_port=q_port,
        threads=4
    )

    prepared_chunks = []
    for idx, doc in enumerate(chunks):
        text = getattr(doc, "page_content", str(doc))
        meta = dict(getattr(doc, "metadata", {}))
        meta["chunk_index"] = idx
        c_id = generate_deterministic_chunk_id(text, meta)
        prepared_chunks.append({"id": c_id, "text": text, "metadata": meta})

    indexer.index_chunks(prepared_chunks, batch_size=32, show_progress=False)

    retriever = ZeroVRAMRetriever(
        qdrant_host=q_host,
        qdrant_port=q_port,
        collection_name=collection_name,
        threads=4
    )
    successful_sources = set()
    for doc in data:
        src = getattr(doc, "metadata", {}).get("source")
        if src:
            successful_sources.add(str(src))
    retriever.successful_files = list(successful_sources)
    return text_splitter, None, retriever


def load_embeddings(model_embedding="qwen2.5:7b-fenced", persist_directory=None, chunk_size=500, chunk_overlap=100):
    """
    Initializes ZeroVRAMRetriever targeting local or containerized Qdrant.
    """
    from scripts.context_assembler import ZeroVRAMRetriever

    q_host = os.getenv("QDRANT_HOST", "qdrant" if os.path.exists("/.dockerenv") else "localhost")
    q_port = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name = os.getenv("QDRANT_COLLECTION", "demo_collection")

    retriever = ZeroVRAMRetriever(
        qdrant_host=q_host,
        qdrant_port=q_port,
        collection_name=collection_name,
        threads=4
    )

    text_splitter = None
    if RecursiveCharacterTextSplitter is not None:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
   
    return text_splitter, None, retriever


def combine_docs(docs):
    return "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)


from scripts.crag_graph import (
    AgentState,
    build_crag_graph as _core_build_crag_graph,
    run_crag_agent as _core_run_crag_agent,
    rag_chain as _core_rag_chain,
)


def build_crag_graph(retriever: Any, model_embedding: str, max_retries: int = 2):
    """
    Constructs CRAG state machine by delegating to scripts.crag_graph.
    Dynamically forwards grader/rewriter/router/llm callbacks to honor test monkeypatches.
    """
    import sys
    this_mod = sys.modules[__name__]
    return _core_build_crag_graph(
        retriever=retriever,
        model_embedding=model_embedding,
        max_retries=max_retries,
        grader_fn=getattr(this_mod, "grade_documents", None),
        rewriter_fn=getattr(this_mod, "rewrite_query", None),
        router_fn=getattr(this_mod, "route_query", None),
        llm_fn=getattr(this_mod, "ollama_llm", None),
    )


def run_crag_agent(
    question: str,
    retriever: Any,
    model_embedding: str,
    max_retries: int = 2
) -> Tuple[str, List[str]]:
    """
    Executes self-correcting CRAG graph by delegating to scripts.crag_graph.
    Dynamically forwards grader/rewriter/router/llm callbacks to honor test monkeypatches.
    """
    import sys
    this_mod = sys.modules[__name__]
    return _core_run_crag_agent(
        question=question,
        retriever=retriever,
        model_embedding=model_embedding,
        max_retries=max_retries,
        grader_fn=getattr(this_mod, "grade_documents", None),
        rewriter_fn=getattr(this_mod, "rewrite_query", None),
        router_fn=getattr(this_mod, "route_query", None),
        llm_fn=getattr(this_mod, "ollama_llm", None),
    )


def rag_chain(question, text_splitter, retriever, model_embedding):
    """
    CRAG pipeline entrypoint preserving polymorphic caller contracts.
    """
    import sys
    this_mod = sys.modules[__name__]
    return _core_rag_chain(
        question,
        text_splitter,
        retriever,
        model_embedding,
        llm_fn=getattr(this_mod, "ollama_llm", None),
    )




def ask_question(
    pdf_bytes: Any,
    question: str,
    create_embeddings: bool = False,
    reset_vectorstore: bool = False,
    embeddings_directory: Optional[str] = None,
    model_embedding: str = "qwen2.5:7b-fenced",
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> str:
    if not question or not str(question).strip():
        return "Error: Question cannot be empty."

    if create_embeddings is True:
        print('create_embeddings')
        if not pdf_bytes:
            return "Error: 'create embeddings' is enabled, but no PDF files or directory were supplied."
        try:
            text_splitter, vectorstore, retriever = process_pdf(
                pdf_bytes,
                model_embedding,
                embeddings_directory,
                chunk_size,
                chunk_overlap,
                reset_existing=reset_vectorstore
            )
        except Exception as e:
            return f"Error during document ingestion: {type(e).__name__}: {str(e)}"

        if retriever is None:
            return "Error: No valid documents were processed from the input."
    else:
        try:
            text_splitter, vectorstore, retriever = load_embeddings(
                model_embedding=model_embedding,
                persist_directory=embeddings_directory,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        except Exception as e:
            return f"Error connecting to vector store: {type(e).__name__}: {str(e)}"

    if retriever is None:
        return "Error: Could not initialize vector retriever."

    result = rag_chain(
        question, 
        text_splitter,  
        retriever, 
        model_embedding
    )
    return result


def process_ingestion(
    pdf_files: Optional[List[Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    reset_vectorstore: bool = False,
    progress: gr.Progress = gr.Progress()
) -> Tuple[str, str]:
    """
    Decoupled ingestion handler with real-time micro-batch progress tracking.
    Processes each file independently: load → chunk → micro-batch index.
    Each 32-chunk batch yields to Gradio's FastAPI event loop via progress.tqdm,
    pushing granular WebSocket updates instead of freezing at 0% until completion.
    Returns (status_message, db_status) tuple for dual UI output binding.
    """
    if not pdf_files:
        return "Error: No PDF files uploaded for ingestion.", get_db_status()

    progress(0, desc="Validating files...")

    # 1. Pre-flight 0-byte filtering & validation
    valid_files = []
    skipped_files = []

    for item in pdf_files:
        file_path = item.name if hasattr(item, 'name') else str(item)
        if not os.path.exists(file_path):
            skipped_files.append((os.path.basename(file_path), "File not found"))
            continue
        try:
            size = os.path.getsize(file_path)
            if size == 0:
                print(f"⚠️ Pre-flight check: Skipping 0-byte file: {file_path}")
                skipped_files.append((os.path.basename(file_path), "0.0 B empty file"))
                continue
            valid_files.append(item)
        except OSError as e:
            skipped_files.append((os.path.basename(file_path), str(e)))

    if not valid_files:
        msg = "Error: All uploaded files were empty (0 bytes) or inaccessible."
        if skipped_files:
            msg += f" Skipped: {', '.join(f'{f} ({r})' for f, r in skipped_files)}"
        return msg, get_db_status()

    # 2. Reset collection if requested (before indexer init)
    if reset_vectorstore:
        progress(0.01, desc="🧹 Wiping existing vector store...")
        clear_db()

    q_host = os.getenv("QDRANT_HOST", "qdrant" if os.path.exists("/.dockerenv") else "localhost")
    q_port = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name = os.getenv("QDRANT_COLLECTION", "demo_collection")
    total = len(valid_files)
    processed_count = 0
    successful_sources: set = set()
    temp_files_to_clean = []

    try:
        from scripts.ingestion_core import QdrantIndexer, generate_deterministic_chunk_id

        # Single shared indexer for the whole session (connection reuse)
        indexer = QdrantIndexer(
            collection_name=collection_name,
            qdrant_host=q_host,
            qdrant_port=q_port,
            threads=4
        )

        # 3. Per-file load → chunk → micro-batch index with fractional multi-stage progress
        for i, item in enumerate(valid_files):
            file_path = item.name if hasattr(item, 'name') else str(item)
            fname = Path(file_path).name
            file_weight = 1.0 / total
            base_prog = i * file_weight
            file_fraction_end = (i + 1) * file_weight

            if is_gradio_artifact(file_path):
                temp_files_to_clean.append(os.path.abspath(file_path))

            try:
                # --- STAGE 1: I/O Loading (10% of file slice) ---
                progress(base_prog + (0.05 * file_weight), desc=f"📄 Stage 1/3: Reading {fname} ({i + 1}/{total})...")

                if PyPDFLoader is None:
                    print(f"⚠️ PyPDFLoader unavailable, skipping {fname}")
                    skipped_files.append((fname, "PyPDFLoader not installed"))
                    continue

                try:
                    loader = PyPDFLoader(file_path=file_path)
                    data = loader.load()
                except Exception as load_err:
                    print(f"⚠️ Load error on {fname}: {load_err}")
                    skipped_files.append((fname, str(load_err)))
                    continue

                if not data:
                    print(f"⚠️ No pages extracted from {fname}")
                    skipped_files.append((fname, "No pages extracted"))
                    continue

                # --- STAGE 2: Splitting & Preparation (20% of file slice) ---
                progress(base_prog + (0.15 * file_weight), desc=f"✂️ Stage 2/3: Chunking {fname} ({i + 1}/{total})...")

                if RecursiveCharacterTextSplitter is not None:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap)
                    )
                    chunks = splitter.split_documents(data)
                else:
                    chunks = data

                # SRE guard: image-only / scanned PDFs yield zero text chunks
                if not chunks:
                    print(f"⚠️ No text chunks from {fname} (image-only PDF?)")
                    skipped_files.append((fname, "No extractable text"))
                    continue

                # Prepare chunk dicts
                prepared_chunks = []
                for idx, doc in enumerate(chunks):
                    text = getattr(doc, "page_content", str(doc))
                    meta = dict(getattr(doc, "metadata", {}))
                    meta["chunk_index"] = idx
                    meta["source_file"] = fname
                    c_id = generate_deterministic_chunk_id(text, meta)
                    prepared_chunks.append({"id": c_id, "text": text, "metadata": meta})

                # --- STAGE 3: Embedding & Indexing (70% of file slice) ---
                batch_size = 32
                total_batches = max(1, (len(prepared_chunks) + batch_size - 1) // batch_size)

                for b_idx in range(total_batches):
                    batch_fraction = (b_idx / total_batches) * (0.70 * file_weight)
                    current_prog = base_prog + (0.30 * file_weight) + batch_fraction
                    progress(
                        current_prog,
                        desc=f"🧠 Stage 3/3: Embedding {fname} ({b_idx + 1}/{total_batches})"
                    )

                    start_idx = b_idx * batch_size
                    batch = prepared_chunks[start_idx : start_idx + batch_size]
                    indexer.index_chunks(batch, batch_size=len(batch), show_progress=False)

                # Track successfully indexed sources
                for doc in data:
                    src = getattr(doc, "metadata", {}).get("source")
                    if src:
                        successful_sources.add(str(src))

                processed_count += 1
                progress(file_fraction_end, desc=f"✅ Done: {fname} ({i + 1}/{total})")

            except Exception as e:
                print(f"⚠️ Failed to process {fname}: {type(e).__name__}: {e}")
                skipped_files.append((fname, str(e)))

    finally:
        # SRE Guard: unconditional temp artifact cleanup regardless of errors
        cleanup_temporary_files(temp_files_to_clean)

    progress(1.0, desc="Complete!")

    if processed_count == 0:
        err = "Error: Document parsing yielded no valid text chunks to index."
        if skipped_files:
            err += f" Skipped: {', '.join(f[0] for f in skipped_files)}"
        return err, get_db_status()

    status_msg = f"✅ Ingestion complete: {processed_count}/{total} file(s) indexed into Qdrant."
    if skipped_files:
        status_msg += f" (Skipped {len(skipped_files)}: {', '.join(f[0] for f in skipped_files)})"
    return status_msg, get_db_status()



def process_query(question: Any = "", *args, **kwargs) -> str:
    """
    Decoupled chat query handler.
    Strictly accepts user text query and dispatches to CRAG state machine.
    Maintains polymorphic backwards compatibility for legacy callers.
    """
    # Handle legacy positional signature: process_query(pdf_files, question, ...)
    if isinstance(question, (list, tuple)) and args:
        question = args[0]
    elif not isinstance(question, str):
        question = str(question) if question else ""

    if not question or not str(question).strip():
        return "Error: Question cannot be empty."

    q_host = os.getenv("QDRANT_HOST", "qdrant" if os.path.exists("/.dockerenv") else "localhost")
    q_port = os.getenv("QDRANT_PORT", "6333")
    embeddings_endpoint = os.getenv("QDRANT_URL", f"http://{q_host}:{q_port}")
    model_name = "qwen2.5:7b-fenced"

    return ask_question(
        pdf_bytes=None,
        question=question,
        create_embeddings=False,
        reset_vectorstore=False,
        embeddings_directory=embeddings_endpoint,
        model_embedding=model_name
    )



def get_db_status() -> str:
    """Queries Qdrant to provide a live status of the vector database (Version-Agnostic)."""
    q_host = os.getenv("QDRANT_HOST", "qdrant" if os.path.exists("/.dockerenv") else "localhost")
    q_port = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name = os.getenv("QDRANT_COLLECTION", "demo_collection")
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=q_host, port=q_port, timeout=3.0)

        collections_response = client.get_collections()
        collection_exists = any(c.name == collection_name for c in collections_response.collections)

        if collection_exists:
            info = client.get_collection(collection_name)
            count = getattr(info, "points_count", getattr(info, "vectors_count", 0)) or 0
            return f"🟢 Ready: Collection '{collection_name}' has {count:,} vectors loaded."
        return "🟡 Empty: No collection found. Ready for first ingestion."
    except Exception as e:
        return f"🔴 Unreachable: Qdrant at {q_host}:{q_port} ({type(e).__name__}: {str(e)})"


def reset_database() -> str:
    """Wipes the Qdrant collection for a clean slate (Version-Agnostic)."""
    q_host = os.getenv("QDRANT_HOST", "qdrant" if os.path.exists("/.dockerenv") else "localhost")
    q_port = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name = os.getenv("QDRANT_COLLECTION", "demo_collection")
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=q_host, port=q_port, timeout=5.0)

        collections_response = client.get_collections()
        collection_exists = any(c.name == collection_name for c in collections_response.collections)

        if collection_exists:
            client.delete_collection(collection_name)
            return f"🟡 Clean Slate: Collection '{collection_name}' wiped successfully."
        return "ℹ️ Info: Collection was already empty."
    except Exception as e:
        return f"🔴 Error resetting database: {type(e).__name__}: {str(e)}"


# Backwards compatibility alias for UI and ingestion callers
clear_db = reset_database


def get_infra_banner() -> str:
    """
    Returns a high-visibility HTML error banner if Qdrant is unreachable.
    Returns empty string when healthy (renders nothing).
    Called via demo.load() — never blocks UI render at startup.
    """
    status = get_db_status()
    if "🔴" in status:
        return (
            '<div style="background:#ffebee;color:#c62828;padding:15px;margin-bottom:12px;'
            'border-radius:6px;border:2px solid #ef5350;">'
            '<h3 style="margin:0 0 8px 0;">⚠️ CRITICAL: Vector Database Offline</h3>'
            f'<p style="margin:0 0 6px 0;"><strong>{status}</strong></p>'
            "<p style=\"margin:0;\">Document uploads will fail. "
            "Ensure the <code>qdrant</code> container is running: "
            "<code>docker compose ps</code></p>"
            "</div>"
        )
    return ""


def create_ui():
    """
    Constructs decoupled industrial-grade Gradio Blocks interface.
    Delegates presentation to ui.gradio_ui while injecting composition root handlers.
    """
    from ui.gradio_ui import create_ui as _build_ui
    return _build_ui(
        query_fn=process_query,
        ingest_fn=process_ingestion,
        db_status_fn=get_db_status,
        clear_db_fn=clear_db,
        banner_fn=get_infra_banner
    )


def main():
    parser = argparse.ArgumentParser(description="RAG Application CLI")
    parser.add_argument("--ingest", action="store_true", help="Ingest PDFs from a directory")
    parser.add_argument("--pdf_path", type=str, help="Path to the PDF directory")
    parser.add_argument("--persist_dir", type=str, default="/datasets/qdrant", help="Directory to persist embeddings")
    parser.add_argument("--model", type=str, default="qwen2.5:7b-fenced", help="Embedding model name")
    parser.add_argument("--chunk_size", type=int, default=500, help="Chunk size for text splitting")
    parser.add_argument("--chunk_overlap", type=int, default=100, help="Chunk overlap for text splitting")
    parser.add_argument("--reset", action="store_true", default=False, help="Reset / wipe persist directory before ingesting")
    parser.add_argument("--share", action="store_true", default=None, help="Launch Gradio with public shareable link")
    parser.add_argument("--server_port", type=int, default=None, help="Gradio server port (defaults to GRADIO_SERVER_PORT or 7860)")
    parser.add_argument("--server_name", type=str, default=None, help="Gradio server host (defaults to GRADIO_SERVER_NAME or 0.0.0.0)")
    
    args = parser.parse_args()

    if args.ingest:
        if not args.pdf_path:
            print("Error: --pdf_path is required for ingestion.")
            return
        print(f"Starting ingestion from {args.pdf_path} (reset={args.reset})...")
        process_pdf(args.pdf_path, args.model, args.persist_dir, args.chunk_size, args.chunk_overlap, reset_existing=args.reset)
        print("Ingestion complete!")
        return

    # Default: Launch Gradio UI
    print('Loading Gradio UI...')
    demo = create_ui()

    server_name = args.server_name or os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = args.server_port or int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    share_flag = args.share if args.share is not None else (os.getenv("GRADIO_SHARE", "False").lower() in ("true", "1", "yes"))

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share_flag
    )


if __name__ == '__main__':
    main()
