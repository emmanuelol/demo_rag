# Demo RAG: LangChain + Ollama + DeepSeek-R1

This repository provides a complete, containerized RAG (Retrieval-Augmented Generation) pipeline for ingesting and querying PDFs using local Large Language Models with GPU acceleration.

## Architecture

- **Document Ingestion:** Polymorphic ingestion supporting individual PDF uploads (`PyPDFLoader`) and directory batch ingestion (`PyPDFDirectoryLoader`).
- **Text Chunking:** `RecursiveCharacterTextSplitter` with configurable chunk size and overlap.
- **Embeddings:** Local embedding generation via `OllamaEmbeddings`.
- **Vector Database:** `Chroma` for on-disk vector persistence with built-in reset/wipe capability to prevent cross-document contamination.
- **Inference Engine:** Local Ollama service executing DeepSeek-R1 reasoning models (`deepseek-r1:1.5b`, `deepseek-r1:7b`) with automated `<think>` block stripping.
- **Resilience / SRE:** Timeout controls (`OLLAMA_TIMEOUT`) and circuit breakers on LLM calls and vector retrieval to prevent UI lockups and thread crashes.

---

## Getting Started

### 1. Prerequisites
- Docker and Docker Compose
- NVIDIA Container Toolkit (for GPU acceleration)

### 2. Configuration
Copy the example environment file and adjust configuration variables:
```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `DATASETS_DIR` | `./datasets` | Host directory containing PDF documents |
| `MODELS_DIR` | `./models` | Host directory for persistent model storage |
| `GRADIO_PORT` | `7860` | Host port exposed for Gradio web interface |
| `GRADIO_SERVER_PORT` | `7860` | Internal container port for Gradio |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Gradio bind address |
| `GRADIO_SHARE` | `false` | Enable/disable public `gradio.live` tunneling |
| `OLLAMA_MODELS` | `deepseek-r1:1.5b` | Space-separated list of models to pull on startup (e.g. `deepseek-r1:1.5b deepseek-r1:7b`) |
| `OLLAMA_TIMEOUT` | `60.0` | Client timeout in seconds for Ollama requests |
| `JUPYTER_PORT` | `8888` | Port for optional Jupyter Lab development container |

### 3. Launching Services
Run the entire stack with Docker Compose:
```bash
# Optional: customize port or models before running
export GRADIO_PORT=7861
export OLLAMA_MODELS="deepseek-r1:1.5b deepseek-r1:7b"

docker compose up -d
```

---

## Accessing the Services
- **RAG Web Interface (Gradio):** http://localhost:7860 (or configured `$GRADIO_PORT`)
- **Development Environment (Jupyter Lab):** http://localhost:8888/lab
- **Ollama API:** http://localhost:11434

---

## Usage Modes

### 1. Interactive Web Interface (Gradio)
Launch the application and navigate to the Gradio web UI:
- **Upload PDF file(s):** Upload one or multiple PDFs directly from your browser.
- **Create embeddings:** Check to ingest uploaded files into the vector store.
- **Reset vector store:** Check to wipe previous embeddings before ingesting new files, eliminating cross-document contamination between distinct runs.
- **Model selection:** Choose between `deepseek-r1:1.5b` and `deepseek-r1:7b`.
- **Chunk size & overlap:** Interactively tune document splitting parameters.

### 2. CLI Batch Ingestion
Execute document ingestion directly via CLI without launching the web server:
```bash
python run_rag.py --ingest \
  --pdf_path /datasets/packt \
  --persist_dir /datasets/deepseek-r1 \
  --model deepseek-r1:1.5b \
  --chunk_size 500 \
  --chunk_overlap 100 \
  --reset
```

#### CLI Options
- `--ingest`: Run batch ingestion mode.
- `--pdf_path`: Path to directory of PDFs or individual PDF file.
- `--persist_dir`: Target directory for Chroma vector store (default: `/datasets/deepseek-r1`).
- `--model`: Model tag for embeddings and inference (default: `deepseek-r1:1.5b`).
- `--chunk_size`: Chunk size in characters (default: `500`).
- `--chunk_overlap`: Chunk overlap in characters (default: `100`).
- `--reset`: Wipe existing persist directory before ingestion.
- `--share`: Launch Gradio with public shareable link.
- `--server_port`: Override Gradio web server port.
- `--server_name`: Override Gradio server bind address.

---

## Project Structure
```text
demo_rag/
├── client/              # Frontend and RAG container definition
│   ├── Dockerfile
│   └── requirements.txt
├── GCP/                 # Google Cloud Platform configuration
├── datasets/            # Default location for PDFs (configurable in .env)
├── models/              # Local storage for LLM weights (configurable in .env)
├── research/            # Experimental notebooks and debug scripts
├── scripts/             # Environment & repository management scripts
├── config.yaml          # Default configuration parameters
├── docker-compose.yaml  # Unified orchestration with GPU allocation
├── entrypoint.sh        # Ollama startup and model provisioning
├── entrypoint_rag.sh    # Client startup script
└── run_rag.py           # Main RAG application entry point (CLI + Gradio)
```

---

## Features
- **Environment Agnostic:** Native Linux and WSL2 ready.
- **GPU Accelerated:** Configured for NVIDIA GPUs with automatic VRAM layer offloading.
- **Vector Isolation:** Safe reset prevents cross-document hallucinations when switching datasets.
- **Network Resilience:** Structured error boundaries protect UI from Ollama timeouts or dropped connections.
- **Secure by Default:** Gradio WAN tunnel (`share`) disabled unless explicitly commanded.

---

## Collaborators
- Emmanuel Ortiz Lopez (https://github.com/emmanuelol)
- Carlos Armando Ortiz Lopez (https://github.com/carlosaol)

