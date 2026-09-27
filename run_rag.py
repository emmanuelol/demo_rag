# Libraries
import argparse
import os
import re
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





def process_pdf(pdf_source, model_embedding, persist_directory, chunk_size, chunk_overlap, reset_existing=False):
    if not pdf_source:
        return None, None, None

    base_url = os.getenv('BASE_URL', 'http://ollama:11434')

    data = []
    if isinstance(pdf_source, list):
        for item in pdf_source:
            file_path = item.name if hasattr(item, 'name') else str(item)
            if os.path.exists(file_path):
                loader = PyPDFLoader(file_path=file_path)
                data.extend(loader.load())
    elif isinstance(pdf_source, str):
        if os.path.isdir(pdf_source):
            loader = PyPDFDirectoryLoader(path=pdf_source)
            data = loader.load()
        elif os.path.isfile(pdf_source):
            loader = PyPDFLoader(file_path=pdf_source)
            data = loader.load()
        else:
            raise FileNotFoundError(f"Path not found: {pdf_source}")
    else:
        raise TypeError(f"Unsupported pdf_source type: {type(pdf_source)}")

    if not data:
        return None, None, None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(data)

    if reset_existing and os.path.exists(persist_directory):
        print(f"Resetting vector store at {persist_directory}...")
        import shutil
        shutil.rmtree(persist_directory, ignore_errors=True)
        os.makedirs(persist_directory, exist_ok=True)

    embeddings = OllamaEmbeddings(model=model_embedding, base_url=base_url)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)
    retriever = vectorstore.as_retriever()

    return text_splitter, vectorstore, retriever


def load_embeddings(model_embedding, persist_directory, chunk_size, chunk_overlap):
    base_url = os.getenv('BASE_URL', 'http://ollama:11434')
    print(f"Loading embeddings for model: {model_embedding}")
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=OllamaEmbeddings(model=model_embedding, base_url=base_url)
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
   
    retriever = vectordb.as_retriever()
    return text_splitter, vectordb, retriever


def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def ollama_llm(question, context, model_embedding):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    base_url = os.getenv('BASE_URL', 'http://ollama:11434')
    timeout = float(os.getenv('OLLAMA_TIMEOUT', '60.0'))
    num_ctx = int(os.getenv('OLLAMA_NUM_CTX', '4096'))
    temperature = float(os.getenv('OLLAMA_TEMPERATURE', '0.2'))

    try:
        client = Client(host=base_url, timeout=timeout)
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
        ans = dispatch_llm_generation(state["question"], context="", model_embedding=model_embedding)
        return {"generation": ans, "trace": state.get("trace", []) + ["direct_chat"]}

    def codebase_ast_node(state: AgentState) -> dict:
        summary_ctx = (
            "Repository architecture comprises modular services: "
            "Semantic Router (scripts/router.py), Retrieval Grader (scripts/grader.py), "
            "ZeroVRAMRetriever (scripts/context_assembler.py), Provider Factory (scripts/factory.py), "
            "and LangGraph State Machine (run_rag.py)."
        )
        ans = dispatch_llm_generation(state["question"], context=summary_ctx, model_embedding=model_embedding)
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
        ans = dispatch_llm_generation(state["question"], formatted_content, model_embedding)
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
        return dispatch_llm_generation(question, "", model_embedding), ["route:general_chat", "direct_chat"]
    if route == "codebase_ast":
        return dispatch_llm_generation(question, "Codebase architecture map", model_embedding), ["route:codebase_ast", "codebase_ast"]

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
            ans = dispatch_llm_generation(question, formatted_content, model_embedding)
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
        print('preloaded')
        if not embeddings_directory or not os.path.exists(embeddings_directory):
            return f"Error: Embeddings directory '{embeddings_directory}' does not exist. Enable 'create embeddings' or provide valid path."
        try:
            text_splitter, vectorstore, retriever = load_embeddings(
                model_embedding,
                embeddings_directory,
                chunk_size,
                chunk_overlap
            )
        except Exception as e:
            return f"Error loading embeddings: {type(e).__name__}: {str(e)}"

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
