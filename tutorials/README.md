# 📚 Corrective RAG (CRAG) Progressive Learning Path

Welcome to the hands-on tutorial track. While the root repository serves as a production-grade, hardened microservice reference architecture, this directory is designed for **developers who want to build the system step-by-step from first principles**.

Each notebook follows a 4-stage engineering narrative:
1. **The Problem** → Show an edge case or failure mode breaking a naive implementation.
2. **The Naive Fix** → The common, brittle approach and why it fails in production.
3. **The Architectural Fix** → How we solve it with clean software design (SOLID).
4. **The Challenge** → An exercise for you to implement and verify.

---

## 🗺️ Curriculum Overview

| # | Notebook | Core Concept | Production Counterpart |
|---|---|---|---|
| **01** | [The Limits of Naive RAG](01_naive_rag.ipynb) | Vector similarity failures, conversational hallucinations | Baseline RAG vs CRAG |
| **02** | [Adding the Router](02_adding_the_router.ipynb) | 0ms semantic routing & vector DB protection | [`scripts/router.py`](../scripts/router.py) |
| **03** | [Building the Grader](03_building_the_grader.ipynb) | Binary relevance filtering & context pruning | [`scripts/grader.py`](../scripts/grader.py) |
| **04** | [The CRAG State Machine](04_crag_state_machine.ipynb) | Cyclic graph state transitions with LangGraph | [`scripts/crag_graph.py`](../scripts/crag_graph.py) |
| **05** | [Hardware-Aware Fencing](05_hardware_fencing.ipynb) | Asymmetric resource allocation & zero-VRAM bleed | [`docker-compose.yaml`](../docker-compose.yaml) & [`scripts/factory.py`](../scripts/factory.py) |
| **06** | [Production Hardening](06_production_hardening.ipynb) | Ragas offline evaluation, Docker mesh, CI/CD gates | [`.github/workflows/rag_eval.yml`](../.github/workflows/rag_eval.yml) & [`tests/`](../tests/) |

---

## 🚀 Getting Started

You can run these notebooks locally using the pre-configured Jupyter Lab container included in the repository stack:

```bash
# 1. Start the container mesh
docker compose up -d

# 2. Open Jupyter Lab in your browser:
# http://localhost:8888
```

Alternatively, run locally with your existing virtual environment:
```bash
pip install -r client/requirements.txt jupyterlab faiss-cpu
jupyter lab
```
