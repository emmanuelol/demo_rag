#!/usr/bin/env python3
"""
utils/ui_helpers.py — Shared UI Utility Functions.

SOLID: Single Responsibility Principle.
    Owns Gradio temp-file lifecycle management.

Bug Fix: Eliminates duplicate _is_gradio_artifact() from run_rag.py.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


_TEMP_DIR: str = os.path.abspath(tempfile.gettempdir())


def is_gradio_artifact(path_str: str) -> bool:
    """
    Returns True if path is Gradio temporary upload artifact.
    """
    norm = os.path.abspath(path_str)
    parts = [p.lower() for p in Path(norm).parts]
    return (
        ("gradio" in parts or ".gradio" in parts)
        and (norm.startswith(_TEMP_DIR) or ".gradio" in parts)
    )


def cleanup_temporary_files(file_paths: list) -> None:
    """
    Safely deletes Gradio temporary PDF artifacts and empty parent directories.
    """
    for path in file_paths:
        try:
            norm_path = os.path.abspath(path)
            parts = [p.lower() for p in Path(norm_path).parts]
            in_temp = norm_path.startswith(_TEMP_DIR) or ".gradio" in parts
            is_grad = "gradio" in parts or ".gradio" in parts

            if in_temp and is_grad and os.path.exists(norm_path) and os.path.isfile(norm_path):
                parent_dir = os.path.dirname(norm_path)
                os.unlink(norm_path)
                parent_parts = [p.lower() for p in Path(parent_dir).parts]
                if (
                    parent_dir.startswith(_TEMP_DIR)
                    and parent_dir != _TEMP_DIR
                    and ("gradio" in parent_parts or ".gradio" in parent_parts)
                ):
                    try:
                        if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                            os.rmdir(parent_dir)
                    except OSError:
                        pass
        except Exception as e:
            print(f"⚠️ Failed to clean up temp file {path}: {e}")
