# Libraries
import argparse
import os
import re
import tempfile
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

    if isinstance(pdf_source, list):
        for item in pdf_source:
            file_path = item.name if hasattr(item, 'name') else str(item)
            if os.path.exists(file_path):
                if PyPDFLoader is not None:
                    loader = PyPDFLoader(file_path=file_path)
                    data.extend(loader.load())
                try:
                    norm_path = os.path.abspath(file_path)
                    if norm_path.startswith(temp_dir) or "gradio" in norm_path.lower():
                        temp_files_to_clean.append(norm_path)
                except Exception:
                    pass
    elif isinstance(pdf_source, str):
        if os.path.isdir(pdf_source):
            if PyPDFDirectoryLoader is not None:
                loader = PyPDFDirectoryLoader(path=pdf_source)
                data.extend(loader.load())
        elif os.path.isfile(pdf_source):
            if PyPDFLoader is not None:
                loader = PyPDFLoader(file_path=pdf_source)
                data.extend(loader.load())
            try:
                norm_path = os.path.abspath(pdf_source)
                if norm_path.startswith(temp_dir) and "gradio" in norm_path.lower():
                    temp_files_to_clean.append(norm_path)
            except Exception:
                pass
        else:
            raise FileNotFoundError(f"Path not found: {pdf_source}")
    else:
        raise TypeError(f"Unsupported pdf_source type: {type(pdf_source)}")

    # Chaos/SRE Guard: Unlink temporary upload files to prevent disk starvation
    for temp_file in temp_files_to_clean:
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        except Exception as e:
            print(f"⚠️ Failed to clean up temp file {temp_file}: {e}")

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




def ask_question(pdf_bytes, question, create_embeddings, reset_vectorstore, embeddings_directory, model_embedding, chunk_size, chunk_overlap):
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


def main():
    parser = argparse.ArgumentParser(description="RAG Application CLI")
    parser.add_argument("--ingest", action="store_true", help="Ingest PDFs from a directory")
    parser.add_argument("--pdf_path", type=str, help="Path to the PDF directory")
    parser.add_argument("--persist_dir", type=str, default="/datasets/deepseek-r1", help="Directory to persist embeddings")
    parser.add_argument("--model", type=str, default="deepseek-r1:1.5b", help="Embedding model name")
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

    interface = gr.Interface(
        fn=ask_question,
        inputs=[
            gr.Files(label="Upload PDF file(s) (optional)", file_types=[".pdf"]), # path to pdf
            gr.Textbox(label="Ask a question"), # question
            gr.Checkbox(value=False, label='create embeddings', info='check to create embeddings'), # check to create embeddings
            gr.Checkbox(value=False, label='reset vector store', info='wipe existing embeddings before ingesting new documents'), # reset vector store
            gr.Textbox(value='/datasets/deepseek-r1', label='Path to embeddings'), # path of the embeddings
            gr.Dropdown(['deepseek-r1:1.5b', 'deepseek-r1:7b'], value='deepseek-r1:1.5b', label='model name'), ## model_name
            gr.Slider(1, 1000, value=500, label='chunk size'), ## chunk_size
            gr.Slider(1, 200, value=100, label='chunk overlap'), ## chuck_overlap
        ],
        outputs="textbox",
        title="Ask questions about Machine Learning, data science and, MLOps based on Packt PDFs",
        description="Use DeepSeek-R1 to answer your questions about the uploaded PDF documents.",
    )

    server_name = args.server_name or os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = args.server_port or int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    share_flag = args.share if args.share is not None else (os.getenv("GRADIO_SHARE", "False").lower() in ("true", "1", "yes"))

    interface.launch(
        server_name=server_name,
        server_port=server_port,
        share=share_flag
    )


if __name__ == '__main__':
    main()
