# Demo RAG: LangChain + Ollama + DeepSeek-R1

This repository provides a complete, containerized RAG (Retrieval-Augmented Generation) application for querying your own PDFs using local Large Language Models.

## Getting Started

### 1. Prerequisites
- Docker and Docker Compose
- NVIDIA Container Toolkit (for GPU acceleration)

### 2. Configuration
Copy the example environment file and adjust the paths to your local dataset and model directories.
```bash
cp .env.example .env
```
- DATASETS_DIR: Absolute path to your PDF documents.
- MODELS_DIR: Absolute path for persistent model storage.

### 3. Launch
Run the entire stack with a single command:
```bash
docker-compose up -d
```

## Accessing the Services
- RAG Interface (Gradio): http://localhost:7860
- Development Environment (Jupyter Lab): http://localhost:8888/lab
- Ollama API: http://localhost:11434

## Project Structure
```text
demo_rag/
├── client/              # Frontend and RAG logic
├── GCP/                 # Google Cloud Platform configuration
├── datasets/            # Default location for PDFs (configurable in .env)
├── models/              # Local storage for LLM weights (configurable in .env)
├── research/            # Experimental notebooks and debug scripts
├── scripts/             # Archived environment-specific scripts
├── docker-compose.yaml  # Unified orchestration
└── run_rag.py           # Main RAG application entry point
```

## Features
- Environment Agnostic: Works seamlessly on Native Linux and WSL2.
- GPU Accelerated: Pre-configured for NVIDIA hardware.
- Persistent Storage: Models and vector embeddings are stored outside containers.
- Live Development: Jupyter Lab runs alongside the RAG application for rapid prototyping.

## Collaborators
- Emmanuel Ortiz Lopez (https://github.com/emmanuelol)
- Carlos Armando Ortiz Lopez (https://github.com/carlosaol)
