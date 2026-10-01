#!/usr/bin/env python3
"""
ui/gradio_ui.py — Decoupled Gradio Web Interface for Industrial RAG System.

SOLID Principles:
  - SRP (Single Responsibility): Manages ONLY the Gradio Blocks UI layout and presentation.
  - DIP (Dependency Inversion): Ingestion, querying, and DB health handlers are injected.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

try:
    import gradio as gr
except ImportError:
    gr = None


def create_ui(
    query_fn: Optional[Callable] = None,
    ingest_fn: Optional[Callable] = None,
    db_status_fn: Optional[Callable] = None,
    clear_db_fn: Optional[Callable] = None,
    banner_fn: Optional[Callable] = None
) -> Any:
    """
    Constructs decoupled industrial-grade Gradio Blocks interface.
    Segregates Ingestion Zone (ETL) from Chat Zone (CRAG Querying)
    to prevent WebSocket timeout and eliminate hidden state.
    """
    if gr is None:
        raise ImportError("Gradio is not installed.")

    # Fallback to no-op stubs if callbacks are not injected
    _query = query_fn or (lambda q: "Error: Query handler not configured.")
    _ingest = ingest_fn or (lambda *args, **kwargs: ("Error: Ingest handler not configured.", "🔴 Offline"))
    _db_status = db_status_fn or (lambda: "⚪ Not connected")
    _clear_db = clear_db_fn or (lambda: "⚪ Not connected")
    _banner = banner_fn or (lambda: "")

    with gr.Blocks(title="Industrial-Grade Zero-VRAM RAG System") as demo:
        gr.Markdown("# 🛡️ Industrial-Grade Zero-VRAM RAG System")
        gr.Markdown("Enterprise Self-Correcting CRAG engine powered by Qwen-2.5, Qdrant & Zero-VRAM FastEmbed.")

        # ── Pre-flight Infrastructure Banner ──────────────────────────────────
        infra_banner = gr.HTML(value="")

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
        demo.load(fn=_db_status, inputs=[], outputs=db_status)
        demo.load(fn=_banner, inputs=[], outputs=infra_banner)

        clear_db_btn.click(fn=_clear_db, inputs=[], outputs=db_status)

        ingest_btn.click(
            fn=_ingest,
            inputs=[
                pdf_input,
                chunk_size_slider,
                chunk_overlap_slider,
                reset_vectorstore_chk
            ],
            outputs=[ingest_status, db_status]
        )

        submit_btn.click(
            fn=_query,
            inputs=[question_input],
            outputs=output_box
        )
        question_input.submit(
            fn=_query,
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
