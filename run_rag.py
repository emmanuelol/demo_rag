# Libraries
import argparse
import os
import re
from time import sleep
import yaml

import gradio as gr
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama
from ollama import ChatResponse, Client, chat


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

    try:
        client = Client(host=base_url, timeout=timeout)
        response = client.chat(
            model=model_embedding,
            messages=[{"role": "user", "content": formatted_prompt}],
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


def rag_chain(question, text_splitter, retriever, model_embedding):
    if retriever is None:
        return "Error: Retriever is not initialized."
    try:
        retrieved_docs = retriever.invoke(question)
    except Exception as e:
        return f"Error: Vector retrieval failed ({type(e).__name__}: {str(e)})"

    formatted_content = combine_docs(retrieved_docs) if retrieved_docs else ""
    return ollama_llm(question, formatted_content, model_embedding)


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
