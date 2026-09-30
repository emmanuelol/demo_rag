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
    timeout = float(os.getenv('OLLAMA_TIMEOUT', '60.0'))
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


def cleanup_temporary_files(file_paths: list) -> None:
    """
    Chaos/SRE Guard: Actively unlinks temporary uploaded PDF artifacts and prunes
    empty temporary parent directories created by Gradio to prevent disk starvation
    and inode exhaustion over extended testing sessions.
    """
    import tempfile
    temp_dir = os.path.abspath(tempfile.gettempdir())

    for path in file_paths:
        try:
            norm_path = os.path.abspath(str(path))
            path_obj = Path(norm_path)
            parts = [p.lower() for p in path_obj.parts]
            is_gradio_temp = (
                ("gradio" in parts or ".gradio" in parts) and
                (norm_path.startswith(temp_dir) or ".gradio" in parts)
            )
            if is_gradio_temp and os.path.exists(norm_path) and os.path.isfile(norm_path):
                parent_dir = os.path.dirname(norm_path)
                os.unlink(norm_path)
                # Prune parent directory if empty and isolated within gradio temp dir
                parent_parts = [p.lower() for p in Path(parent_dir).parts]
                if parent_dir.startswith(temp_dir) and parent_dir != temp_dir and ("gradio" in parent_parts or ".gradio" in parent_parts):
                    try:
                        if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                            os.rmdir(parent_dir)
                    except OSError:
                        pass
        except Exception as e:
            print(f"⚠️ Failed to clean up temp file {path}: {e}")


def process_pdf(pdf_source, model_embedding, persist_directory, chunk_size, chunk_overlap, reset_existing=False):
    """
    Extracts text from PDF source, chunks with text splitter, and indexes into Qdrant via Zero-VRAM FastEmbed.
    Automatically garbage-collects temporary uploaded PDF artifacts to prevent disk exhaustion.
    """
    if not pdf_source:
        return None, None, None

    data = []
    temp_files_to_clean = []
    import tempfile
    temp_dir = os.path.abspath(tempfile.gettempdir())

    def _is_gradio_artifact(path_str: str) -> bool:
        norm = os.path.abspath(path_str)
        parts = [p.lower() for p in Path(norm).parts]
        return ("gradio" in parts or ".gradio" in parts) and (norm.startswith(temp_dir) or ".gradio" in parts)

    try:
        if isinstance(pdf_source, list):
            for item in pdf_source:
                file_path = item.name if hasattr(item, 'name') else str(item)
                if os.path.exists(file_path):
                    if _is_gradio_artifact(file_path):
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
                if _is_gradio_artifact(pdf_source):
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


class AgentState(TypedDict):
    question: str
    search_query: str
    route: str
    documents: List[Dict[str, Any]]
    is_relevant: bool
    generation: str
    retry_count: int
    max_retries: int
    trace: List[str]


def build_crag_graph(retriever: Any, model_embedding: str, max_retries: int = 2):
    """
    Constructs the Corrective RAG (CRAG) state machine with a hard recursion limit.
    """
    if StateGraph is None:
        return None

    def route_node(state: AgentState) -> dict:
        route = route_query(state["question"], model_name=model_embedding)
        return {"route": route, "trace": state.get("trace", []) + [f"route:{route}"]}

    def direct_chat_node(state: AgentState) -> dict:
        ans = ollama_llm(state["question"], context="", model_embedding=model_embedding)
        return {"generation": ans, "trace": state.get("trace", []) + ["direct_chat"]}

    def codebase_ast_node(state: AgentState) -> dict:
        summary_ctx = (
            "Repository architecture comprises modular services: "
            "Semantic Router (scripts/router.py), Retrieval Grader (scripts/grader.py), "
            "ZeroVRAMRetriever (scripts/context_assembler.py), Provider Factory (scripts/factory.py), "
            "and LangGraph State Machine (run_rag.py)."
        )
        ans = ollama_llm(state["question"], context=summary_ctx, model_embedding=model_embedding)
        return {"generation": ans, "trace": state.get("trace", []) + ["codebase_ast"]}

    def retrieve_node(state: AgentState) -> dict:
        q = state.get("search_query") or state["question"]
        docs = []
        if retriever is not None:
            try:
                if hasattr(retriever, "retrieve_and_rerank"):
                    docs = retriever.retrieve_and_rerank(q, retrieve_limit=20, rerank_top_k=5)
                elif hasattr(retriever, "invoke"):
                    raw = retriever.invoke(q)
                    docs = [{"id": i, "text": d.page_content, "score": 1.0} for i, d in enumerate(raw)]
            except Exception as e:
                print(f"⚠️ Vector retrieval failure: {e}")
                docs = [{"id": "fallback", "text": f"Retrieval error: {e}", "metadata": {"error": str(e)}}]
        return {"documents": docs, "trace": state.get("trace", []) + [f"retrieve:{len(docs)}"]}

    def grade_node(state: AgentState) -> dict:
        docs = state.get("documents", [])
        is_relevant, relevant_docs = grade_documents(state["question"], docs, model_name=model_embedding)
        return {
            "is_relevant": is_relevant,
            "documents": relevant_docs,
            "trace": state.get("trace", []) + [f"grade:{is_relevant}"]
        }

    def rewrite_node(state: AgentState) -> dict:
        retries = state.get("retry_count", 0) + 1
        rewritten = rewrite_query(state["question"], model_name=model_embedding)
        return {
            "search_query": rewritten,
            "retry_count": retries,
            "trace": state.get("trace", []) + [f"rewrite:{retries}"]
        }

    def generate_node(state: AgentState) -> dict:
        docs = state.get("documents", [])
        formatted_content = "\n\n".join(d.get("text", "") for d in docs if d.get("text"))
        ans = ollama_llm(state["question"], formatted_content, model_embedding)
        return {"generation": ans, "trace": state.get("trace", []) + ["generate"]}

    def fallback_node(state: AgentState) -> dict:
        msg = "I could not find sufficient relevant context in the provided documents to answer your question accurately."
        return {"generation": msg, "trace": state.get("trace", []) + ["fallback_exhausted"]}

    builder = StateGraph(AgentState)
    builder.add_node("route_node", route_node)
    builder.add_node("direct_chat_node", direct_chat_node)
    builder.add_node("codebase_ast_node", codebase_ast_node)
    builder.add_node("retrieve_node", retrieve_node)
    builder.add_node("grade_node", grade_node)
    builder.add_node("rewrite_node", rewrite_node)
    builder.add_node("generate_node", generate_node)
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
    retriever: Any,
    model_embedding: str,
    max_retries: int = 2
) -> Tuple[str, List[str]]:
    """
    Executes the self-correcting CRAG graph.
    Returns (generation_text, execution_trace).
    """
    graph = build_crag_graph(retriever, model_embedding, max_retries=max_retries)

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
    route = route_query(question, model_name=model_embedding)
    if route == "general_chat":
        return ollama_llm(question, "", model_embedding), ["route:general_chat", "direct_chat"]
    if route == "codebase_ast":
        return ollama_llm(question, "Codebase architecture map", model_embedding), ["route:codebase_ast", "codebase_ast"]

    if retriever is None:
        return "Error: Retriever is not initialized.", ["route:vector_search", "no_retriever"]

    retries = 0
    search_q = question
    trace = [f"route:{route}"]

    while retries <= max_retries:
        if hasattr(retriever, "retrieve_and_rerank"):
            docs = retriever.retrieve_and_rerank(search_q, retrieve_limit=20, rerank_top_k=5)
        elif hasattr(retriever, "invoke"):
            docs = [{"id": i, "text": d.page_content, "score": 1.0} for i, d in enumerate(retriever.invoke(search_q))]
        else:
            return "Error: Unsupported retriever interface.", trace

        trace.append(f"retrieve:{len(docs)}")
        is_relevant, relevant_docs = grade_documents(question, docs, model_name=model_embedding)
        trace.append(f"grade:{is_relevant}")

        if is_relevant:
            formatted_content = "\n\n".join(d.get("text", "") for d in relevant_docs if d.get("text"))
            ans = ollama_llm(question, formatted_content, model_embedding)
            trace.append("generate")
            return ans, trace

        retries += 1
        if retries <= max_retries:
            search_q = rewrite_query(question, model_name=model_embedding)
            trace.append(f"rewrite:{retries}")

    trace.append("fallback_exhausted")
    return "I could not find sufficient relevant context in the provided documents to answer your question accurately.", trace


def rag_chain(question, text_splitter, retriever, model_embedding):
    """
    CRAG pipeline entrypoint preserving polymorphic caller contracts.
    """
    generation, _ = run_crag_agent(question, retriever, model_embedding, max_retries=2)
    return generation




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
    import tempfile as _tempfile
    temp_dir = os.path.abspath(_tempfile.gettempdir())

    def _is_gradio_artifact(path_str: str) -> bool:
        norm = os.path.abspath(path_str)
        parts = [p.lower() for p in Path(norm).parts]
        return ("gradio" in parts or ".gradio" in parts) and (norm.startswith(temp_dir) or ".gradio" in parts)

    try:
        from scripts.ingestion_core import QdrantIndexer, generate_deterministic_chunk_id

        # Single shared indexer for the whole session (connection reuse)
        indexer = QdrantIndexer(
            collection_name=collection_name,
            qdrant_host=q_host,
            qdrant_port=q_port,
            threads=4
        )

        # 3. Per-file load → chunk → micro-batch index
        for i, item in enumerate(valid_files):
            file_path = item.name if hasattr(item, 'name') else str(item)
            fname = Path(file_path).name
            file_fraction_start = i / total
            file_fraction_end = (i + 1) / total

            progress(file_fraction_start, desc=f"📖 Parsing {fname} ({i + 1}/{total})")

            if _is_gradio_artifact(file_path):
                temp_files_to_clean.append(os.path.abspath(file_path))

            try:
                # Load
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

                # Chunk
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

                # 🚀 Micro-batch index: yields to Gradio WebSocket every 32 chunks
                batch_size = 32
                total_batches = max(1, (len(prepared_chunks) + batch_size - 1) // batch_size)

                for b_idx in progress.tqdm(
                    range(total_batches),
                    desc=f"⚡ Embedding {fname}"
                ):
                    start_idx = b_idx * batch_size
                    batch = prepared_chunks[start_idx: start_idx + batch_size]
                    indexer.index_chunks(batch, batch_size=len(batch), show_progress=False)

                # Track successfully indexed sources
                for doc in data:
                    src = getattr(doc, "metadata", {}).get("source")
                    if src:
                        successful_sources.add(str(src))

                processed_count += 1
                progress(file_fraction_end, desc=f"✅ Done: {fname}")

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
    """
    Queries Qdrant for live collection vector count.
    Returns a human-readable status string safe for display in gr.Textbox.
    Fault-tolerant: never raises — returns degraded status on any connection error.
    """
    q_host = os.getenv("QDRANT_HOST", "qdrant" if os.path.exists("/.dockerenv") else "localhost")
    q_port = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name = os.getenv("QDRANT_COLLECTION", "demo_collection")
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=q_host, port=q_port, timeout=3.0)
        info = client.get_collection(collection_name)
        count = info.vectors_count if info.vectors_count is not None else 0
        return f"🟢 Ready: {count:,} vector(s) in '{collection_name}'"
    except Exception as e:
        err = str(e)
        if "not found" in err.lower() or "doesn't exist" in err.lower() or "404" in err:
            return f"🟡 Empty: Collection '{collection_name}' does not exist yet."
        return f"🔴 Unreachable: Qdrant at {q_host}:{q_port} ({type(e).__name__})"


def clear_db() -> str:
    """
    Atomically wipes the Qdrant collection for a clean slate.
    Uses delete_collection (not filesystem rmtree) because this repo targets
    a networked Qdrant service — there is no local vector folder to wipe.
    """
    q_host = os.getenv("QDRANT_HOST", "qdrant" if os.path.exists("/.dockerenv") else "localhost")
    q_port = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name = os.getenv("QDRANT_COLLECTION", "demo_collection")
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=q_host, port=q_port, timeout=5.0)
        collections = [c.name for c in client.get_collections().collections]
        if collection_name in collections:
            client.delete_collection(collection_name)
        return get_db_status()
    except Exception as e:
        return f"🔴 Clear failed: {type(e).__name__}: {e}"


def create_ui():
    """
    Constructs decoupled industrial-grade Gradio Blocks interface.
    Segregates Ingestion Zone (ETL) from Chat Zone (CRAG Querying)
    to prevent WebSocket timeout and eliminate hidden state.
    """
    if gr is None:
        raise ImportError("Gradio is not installed.")

    with gr.Blocks(title="Industrial-Grade Zero-VRAM RAG System") as demo:
        gr.Markdown("# 🛡️ Industrial-Grade Zero-VRAM RAG System")
        gr.Markdown("Enterprise Self-Correcting CRAG engine powered by Qwen-2.5, Qdrant & Zero-VRAM FastEmbed.")

        # ── DB Status Bar ──────────────────────────────────────────────────────
        with gr.Row():
            db_status = gr.Textbox(
                value="Checking...",
                label="📊 Vector Database Status",
                interactive=False,
                scale=4
            )
            clear_db_btn = gr.Button("🗑️ Clear Vector DB", variant="stop", scale=1)

        with gr.Row():
            # Ingestion Zone (ETL Pipeline)
            with gr.Column(scale=1):
                gr.Markdown("### 📂 Ingestion Zone (ETL Pipeline)")
                pdf_input = gr.Files(
                    label="Upload PDF file(s)",
                    file_types=[".pdf"]
                )
                chunk_size_slider = gr.Slider(
                    minimum=100,
                    maximum=2000,
                    value=500,
                    step=50,
                    label="chunk size"
                )
                chunk_overlap_slider = gr.Slider(
                    minimum=0,
                    maximum=500,
                    value=100,
                    step=10,
                    label="chunk overlap"
                )
                reset_vectorstore_chk = gr.Checkbox(
                    value=False,
                    label="reset vector store",
                    info="Wipe existing embeddings before ingesting new documents"
                )
                ingest_btn = gr.Button("⚙️ Create Vectors & Ingest", variant="secondary")
                ingest_status = gr.Textbox(
                    label="Ingestion Status",
                    lines=3,
                    interactive=False
                )

            # Chat Zone (CRAG Query Engine)
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Chat Zone (CRAG Query Engine)")
                question_input = gr.Textbox(
                    label="Ask a question",
                    placeholder="Enter your question here...",
                    lines=3
                )
                with gr.Row():
                    submit_btn = gr.Button("Submit Query", variant="primary")
                    clear_btn = gr.Button("Clear")
                output_box = gr.Textbox(
                    label="Response",
                    lines=10,
                    interactive=False
                )

        # ── Event Handlers: Segregated Pipelines ───────────────────────────────
        # Populate DB status on page load without blocking UI render
        demo.load(fn=get_db_status, inputs=[], outputs=db_status)

        clear_db_btn.click(fn=clear_db, inputs=[], outputs=db_status)

        ingest_btn.click(
            fn=process_ingestion,
            inputs=[
                pdf_input,
                chunk_size_slider,
                chunk_overlap_slider,
                reset_vectorstore_chk
            ],
            outputs=[ingest_status, db_status]
        )

        submit_btn.click(
            fn=process_query,
            inputs=[question_input],
            outputs=output_box
        )
        question_input.submit(
            fn=process_query,
            inputs=[question_input],
            outputs=output_box
        )
        clear_btn.click(
            fn=lambda: (None, "", "", ""),
            inputs=[],
            outputs=[pdf_input, ingest_status, question_input, output_box]
        )

    demo.queue()
    return demo


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
