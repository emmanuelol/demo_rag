#!/usr/bin/env python3
"""
Context Assembler - Builds targeted LLM context from the repository graph
and provides Zero-VRAM CPU Hybrid Retrieval with FastEmbed, Qdrant, and FlashRank.
Supports multi-level dependency traversal, token budgeting, and impact analysis.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Any, List, Dict

# Explicit CPU thread fences to prevent Ryzen 7 thread starvation
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import yaml

try:
    from fastembed import TextEmbedding
except ImportError:
    TextEmbedding = None

try:
    from flashrank import Ranker, RerankRequest
except ImportError:
    Ranker = None
    class RerankRequest:
        def __init__(self, query: str, passages: list):
            self.query = query
            self.passages = passages

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except ImportError:
    QdrantClient = None
    qmodels = None

REPO_ROOT = Path(__file__).resolve().parent.parent

# === CONFIGURATION ===
DEFAULT_MAX_DEPTH = 2          # How many levels of callers/callees to traverse
DEFAULT_TOKEN_BUDGET = 80000   # Approximate token budget (chars / 4)
CHARS_PER_TOKEN = 4



# ============================================================
# GRAPH LOADING
# ============================================================

def load_graph_data(graph_path: Optional[Path] = None) -> Optional[dict]:
    """Load the complete graph JSON."""
    if graph_path is None:
        graph_path = REPO_ROOT / "local_codegraph.json"

    if not graph_path.exists():
        print(f"❌ Error: Graph not found at {graph_path}")
        print(f"   Run: python scripts/map_repository.py")
        return None

    with open(graph_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# DEPENDENCY TRAVERSAL (Multi-level)
# ============================================================

def find_callees(target_file: str, graph: dict, edges: list, max_depth: int = 2) -> dict:
    """Find all files that target_file depends on (imports), up to max_depth."""
    result = {}
    queue = [(target_file, 0)]
    visited = {target_file}

    adj = {}
    for edge in edges:
        src = edge["from"]
        if src not in adj:
            adj[src] = []
        adj[src].append(edge["to"])

    while queue:
        current, depth = queue.pop(0)
        if depth >= max_depth:
            continue

        for neighbor in adj.get(current, []):
            if neighbor not in visited and neighbor in graph:
                visited.add(neighbor)
                result[neighbor] = depth + 1
                queue.append((neighbor, depth + 1))

    return result


def find_callers(target_file: str, graph: dict, edges: list, max_depth: int = 2) -> dict:
    """Find all files that depend on target_file (reverse deps), up to max_depth."""
    result = {}
    queue = [(target_file, 0)]
    visited = {target_file}

    reverse_adj = {}
    for edge in edges:
        dst = edge["to"]
        if dst not in reverse_adj:
            reverse_adj[dst] = []
        reverse_adj[dst].append(edge["from"])

    while queue:
        current, depth = queue.pop(0)
        if depth >= max_depth:
            continue

        for neighbor in reverse_adj.get(current, []):
            if neighbor not in visited and neighbor in graph:
                visited.add(neighbor)
                result[neighbor] = depth + 1
                queue.append((neighbor, depth + 1))

    return result


def get_impact_analysis(target_file: str, graph: dict, edges: list) -> dict:
    """Analyze what would break if target_file changes."""
    callers = find_callers(target_file, graph, edges, max_depth=3)

    high_impact = []
    medium_impact = []

    for caller, depth in callers.items():
        caller_data = graph.get(caller, {})
        importance = len(caller_data.get("functions", [])) + len(caller_data.get("classes", []))
        entry = {"file": caller, "depth": depth, "complexity": importance}

        if depth == 1:
            high_impact.append(entry)
        else:
            medium_impact.append(entry)

    return {
        "direct_dependents": high_impact,
        "transitive_dependents": medium_impact,
        "total_affected": len(callers)
    }


# ============================================================
# NODE DESCRIPTION GENERATION
# ============================================================

def describe_node(file_path: str, data: dict) -> str:
    """Generate a human-readable description of what a file/node does."""
    lines = []
    node_type = data.get("type", "unknown")

    if node_type == "python":
        doc = data.get("module_docstring", "")
        if doc:
            first_line = doc.strip().splitlines()[0]
            if first_line:
                lines.append(f"  Purpose: {first_line}")

        summary = data.get("summary", "")
        if summary:
            lines.append(f"  Summary: {summary}")

        for cls in data.get("classes", []):
            methods_str = ", ".join(m["name"] for m in cls.get("methods", [])[:8])
            doc_str = f" - {cls['docstring'].strip().splitlines()[0][:80]}" if cls.get("docstring") else ""
            lines.append(f"  Class `{cls['name']}`{doc_str}")
            if methods_str:
                lines.append(f"    Methods: {methods_str}")

        for func in data.get("functions", []):
            sig = _format_function_signature(func)
            doc_str = f" | {func['docstring'].strip().splitlines()[0][:60]}" if func.get("docstring") else ""
            lines.append(f"  Func `{sig}`{doc_str}")

        metrics = data.get("metrics", {})
        if metrics:
            lines.append(f"  Metrics: {metrics.get('line_count', '?')} lines, "
                        f"{metrics.get('total_functions', '?')} functions, "
                        f"complexity={metrics.get('total_complexity', '?')}")

    elif node_type == "docker":
        lines.append(f"  Base: {', '.join(data.get('base_images', ['unknown']))}")
        if data.get("exposed_ports"):
            lines.append(f"  Ports: {', '.join(data['exposed_ports'])}")

    elif node_type == "config_file":
        keys = data.get("top_level_keys", [])
        if keys:
            lines.append(f"  Keys: {', '.join(keys[:10])}")

    elif node_type == "generic_text":
        summary = data.get("summary", "")
        if summary:
            lines.append(f"  Content: {summary[:100]}")

    return "\n".join(lines) if lines else "  (No description available)"


def _format_function_signature(func: dict) -> str:
    """Format a function signature string."""
    name = func.get("name", "?")
    args = func.get("args", [])
    args_str = ", ".join(
        f"{a['name']}{': ' + a['annotation'] if a.get('annotation') else ''}"
        for a in args[:5]
    )
    if len(args) > 5:
        args_str += ", ..."
    ret = f" -> {func['return_annotation']}" if func.get("return_annotation") else ""
    prefix = "async " if func.get("is_async") else ""
    return f"{prefix}{name}({args_str}){ret}"


# ============================================================
# TOKEN BUDGET MANAGEMENT
# ============================================================

def estimate_tokens(text: str) -> int:
    """Rough token estimate."""
    return len(text) // CHARS_PER_TOKEN


def prioritize_files(target_file: str, callers: dict, callees: dict,
                     importance_scores: dict, budget: int) -> list:
    """Order files by relevance, deduplicating and respecting ranking."""
    ordered = [(target_file, 0, 100)]

    for f, depth in sorted(callers.items(), key=lambda x: x[1]):
        priority = 80 - depth * 10 + importance_scores.get(f, 0)
        ordered.append((f, depth, priority))

    for f, depth in sorted(callees.items(), key=lambda x: x[1]):
        priority = 70 - depth * 10 + importance_scores.get(f, 0)
        ordered.append((f, depth, priority))

    ordered.sort(key=lambda x: -x[2])

    seen = set()
    result = []
    for f, _, _ in ordered:
        if f not in seen:
            seen.add(f)
            result.append(f)

    return result


# ============================================================
# CONTEXT BUILDER
# ============================================================

def build_targeted_context(
    target_file: str,
    full_graph_data: dict,
    is_audit_mode: bool = False,
    max_depth: int = DEFAULT_MAX_DEPTH,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    include_source: bool = True,
    output_file: str = "targeted_context.md"
):
    """Assemble a comprehensive context document for LLM consumption."""

    # Normalize target_file path relative to REPO_ROOT
    try:
        p = Path(target_file)
        if p.is_absolute():
            target_file = str(p.relative_to(REPO_ROOT))
        else:
            target_file = str(p).lstrip('./')
    except Exception:
        pass

    metadata = full_graph_data.get("metadata", {})
    graph = full_graph_data.get("files", full_graph_data)
    edges = full_graph_data.get("edges", [])
    importance_scores = full_graph_data.get("importance_scores", {})
    branch_name = metadata.get("branch", "unknown-branch")

    if target_file not in graph:
        print(f"❌ Error: '{target_file}' not found in graph (branch: {branch_name})")
        print(f"   Available files: {len(graph)}")
        suggestions = [f for f in graph if target_file.split('/')[-1] in f][:5]
        if suggestions:
            print(f"   Did you mean: {suggestions}")
        return

    node_data = graph[target_file]
    node_type = node_data.get("type", "unknown")

    print(f"🔍 Analyzing: {target_file}")
    print(f"   Type: {node_type} | Branch: {branch_name}")
    print(f"   Mode: {'AUDIT (QA)' if is_audit_mode else 'PLANNING (SRE)'}")
    print(f"   Depth: {max_depth} | Budget: ~{token_budget // 1000}K tokens")

    callees = find_callees(target_file, graph, edges, max_depth) if node_type == "python" else {}
    callers = find_callers(target_file, graph, edges, max_depth) if node_type == "python" else {}
    impact = get_impact_analysis(target_file, graph, edges)

    print(f"   Found: {len(callers)} callers, {len(callees)} callees, "
          f"{impact['total_affected']} total affected")

    all_files = prioritize_files(target_file, callers, callees, importance_scores, token_budget)

    output_path = REPO_ROOT / output_file
    chars_used = 0

    with open(output_path, 'w', encoding='utf-8') as out:
        out.write("# LLM Context Assembly\n\n")
        out.write(f"**Branch:** `{branch_name}`  \n")
        out.write(f"**Target:** `{target_file}`  \n")
        out.write(f"**Mode:** {'Audit/QA' if is_audit_mode else 'Planning/SRE'}  \n")
        out.write(f"**Generated by:** context_assembler.py v2.0\n\n")

        out.write("---\n\n")
        if is_audit_mode:
            out.write("## SYSTEM INSTRUCTIONS (Audit Mode)\n\n")
            out.write("**ROLE:** Principal QA Engineer & Architecture Auditor\n\n")
            out.write("**OBJECTIVE:** Strict post-implementation review. Verify objectives met, no regressions.\n\n")
            out.write("**RULES:**\n")
            out.write("1. **Regression Profiling:** Did changes break any caller's interface?\n")
            out.write("2. **DoD Verification:** Evaluate against Definition of Done below.\n")
            out.write("3. **Anti-Fragility:** No silent failures, memory leaks, or latent bugs.\n")
            out.write("4. **Branch Context:** Changes must align with branch purpose.\n\n")
            out.write("**EXPECTED OUTPUT:**\n")
            out.write("- (A) Goal Achievement Status [PASS / FAIL / PARTIAL]\n")
            out.write("- (B) Regression Analysis\n")
            out.write("- (C) Residual Risks & Code Smells\n")
            out.write("- (D) Final Verdict (Merge / Hotfix / Reject)\n\n")
            out.write("> ⚠️ **PASTE DEFINITION OF DONE BELOW:**\n")
            out.write("> DoD: [...]\n\n")
        else:
            out.write("## SYSTEM INSTRUCTIONS (Planning Mode)\n\n")
            out.write("**ROLE:** Principal Software Engineer, SRE Architect & Chaos Engineer\n\n")
            out.write("**OBJECTIVE:** Deep code audit and scientific debugging.\n\n")
            out.write("**RULES:**\n")
            out.write("1. **Boundary Analysis:** Cross-reference callers/callees. Don't break contracts.\n")
            out.write("2. **Chaos Engineering:** Assume network fails, DB drops. Ensure idempotency.\n")
            out.write("3. **Zero Quick Patches:** Find root cause. Explain the logical flaw.\n")
            out.write("4. **Branch Alignment:** Fixes must serve the branch's purpose.\n\n")
            out.write("**EXPECTED OUTPUT:**\n")
            out.write("- (A) Architecture Diagnosis\n")
            out.write("- (B) Risks and Errors (Bugs/Blockers)\n")
            out.write("- (C) Execution Action Plan (Task, Target File, Location, Required Action, DoD, Validation Method)\n\n")

        out.write("---\n\n")
        out.write("## Architecture Map\n\n")

        out.write(f"### Target Node: `{target_file}`\n")
        out.write(f"```\n{describe_node(target_file, node_data)}\n```\n\n")

        out.write("### Callers (files that depend on target):\n\n")
        if callers:
            for caller, depth in sorted(callers.items(), key=lambda x: x[1]):
                indent = "  " * depth
                summary = graph.get(caller, {}).get("summary", "")[:80]
                out.write(f"{indent}- `{caller}` (depth={depth}) — {summary}\n")
        else:
            out.write("- None detected.\n")
        out.write("\n")

        out.write("### Callees (files that target depends on):\n\n")
        if callees:
            for callee, depth in sorted(callees.items(), key=lambda x: x[1]):
                indent = "  " * depth
                summary = graph.get(callee, {}).get("summary", "")[:80]
                out.write(f"{indent}- `{callee}` (depth={depth}) — {summary}\n")
        else:
            out.write("- None detected.\n")
        out.write("\n")

        out.write("### Impact Analysis (if this file changes):\n\n")
        out.write(f"- **Direct dependents:** {len(impact['direct_dependents'])}\n")
        out.write(f"- **Transitive dependents:** {len(impact['transitive_dependents'])}\n")
        out.write(f"- **Total affected files:** {impact['total_affected']}\n")
        if impact['direct_dependents']:
            out.write(f"- **High-risk callers:** ")
            out.write(", ".join(f"`{d['file']}`" for d in impact['direct_dependents'][:5]))
            out.write("\n")
        out.write("\n")

        if include_source:
            out.write("---\n\n")
            out.write("## Source Code\n\n")

            for filepath in all_files:
                if filepath not in graph:
                    continue

                source_path = REPO_ROOT / filepath
                try:
                    source = source_path.read_text(encoding='utf-8', errors='replace')
                except Exception as e:
                    source = f"# Error reading file: {e}"

                if filepath != target_file:
                    chars_used += len(source)
                    if chars_used > token_budget * CHARS_PER_TOKEN:
                        out.write(f"\n> ⚠️ Token budget reached. Remaining files omitted.\n")
                        idx = all_files.index(filepath)
                        out.write(f"> Skipped: {', '.join(all_files[idx:])}\n")
                        break
                else:
                    chars_used += len(source)

                md_lang = _get_markdown_language(filepath)
                depth_label = ""
                if filepath in callers:
                    depth_label = f" [CALLER depth={callers[filepath]}]"
                elif filepath in callees:
                    depth_label = f" [CALLEE depth={callees[filepath]}]"
                elif filepath == target_file:
                    depth_label = " [TARGET]"

                out.write(f"### FILE: `{filepath}`{depth_label}\n")

                desc = describe_node(filepath, graph[filepath])
                if desc and filepath != target_file:
                    first_desc = desc.strip().splitlines()[0] if desc.strip() else ""
                    if first_desc:
                        out.write(f"> {first_desc}\n\n")

                out.write(f"```{md_lang}\n{source}\n```\n\n")

    file_size = output_path.stat().st_size
    print(f"\n✅ Context generated: {output_path}")
    print(f"   Size: {file_size // 1024}KB (~{file_size // CHARS_PER_TOKEN} tokens)")
    print(f"   Files included: {len(all_files)}")


def _get_markdown_language(filepath: str) -> str:
    """Get markdown code fence language."""
    ext = Path(filepath).suffix.lower()
    name = Path(filepath).name.lower()

    if 'dockerfile' in name:
        return 'dockerfile'
    if 'makefile' in name:
        return 'makefile'

    return {
        '.py': 'python', '.json': 'json', '.yml': 'yaml', '.yaml': 'yaml',
        '.md': 'markdown', '.sh': 'bash', '.js': 'javascript', '.ts': 'typescript',
        '.html': 'html', '.css': 'css', '.sql': 'sql', '.toml': 'toml',
        '.cfg': 'ini', '.ini': 'ini', '.txt': 'text'
    }.get(ext, '')


# ============================================================
# ZERO-VRAM HYBRID RETRIEVAL (CPU FastEmbed + Qdrant + FlashRank)
# ============================================================

def parse_memory_frontmatter(content: str) -> dict:
    """
    Parses YAML frontmatter from markdown documents (e.g. memory.md).
    Raises ValueError on malformed or unclosed frontmatter delimiters.
    """
    if not content or not content.strip().startswith("---"):
        return {"metadata": {}, "body": content or ""}

    lines = content.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {"metadata": {}, "body": content}

    closing_index = -1
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing_index = idx
            break

    if closing_index == -1:
        raise ValueError("Malformed YAML frontmatter: unclosed delimiter '---'")

    frontmatter_raw = "".join(lines[1:closing_index])
    body = "".join(lines[closing_index + 1:]).lstrip("\n")

    try:
        metadata = yaml.safe_load(frontmatter_raw)
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(f"Malformed YAML frontmatter: expected dict mapping, got {type(metadata).__name__}")
        return {"metadata": metadata or {}, "body": body}
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed YAML frontmatter: {exc}") from exc


class ZeroVRAMRetriever:
    """
    Zero-VRAM Hybrid Retrieval Engine.
    Executes dense embeddings and reranking exclusively on host CPU (capped at threads=4)
    and queries Qdrant Vector DB (bounded at 4GB RAM) with idempotency and fallback handling.
    """
    def __init__(
        self,
        qdrant_host: str = "qdrant",
        qdrant_port: int = 6333,
        collection_name: str = "demo_rag_collection",
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        reranker_model_name: str = "ms-marco-TinyBERT-L-2-v2",
        threads: int = 4,
        client: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
        reranker: Optional[Any] = None
    ):
        self.qdrant_host = os.getenv("QDRANT_HOST", qdrant_host)
        self.qdrant_port = int(os.getenv("QDRANT_PORT", str(qdrant_port)))
        self.collection_name = collection_name
        self.threads = threads

        if client is not None:
            self.client = client
        elif QdrantClient is not None:
            try:
                self.client = QdrantClient(
                    host=self.qdrant_host,
                    port=self.qdrant_port,
                    timeout=5.0,
                    check_compatibility=False
                )
            except Exception as e:
                print(f"⚠️ Qdrant client connection failed: {e}")
                self.client = None
        else:
            self.client = None

        # Pre-flight auto-creation to avoid cold-start 404 errors
        self._ensure_collection_exists()

        if embedding_model is not None:
            self.embedding_model = embedding_model
        elif TextEmbedding is not None:
            try:
                self.embedding_model = TextEmbedding(model_name=embedding_model_name, threads=threads)
            except Exception as e:
                print(f"⚠️ FastEmbed initialization warning: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None

        if reranker is not None:
            self.reranker = reranker
        elif Ranker is not None:
            try:
                self.reranker = Ranker(model_name=reranker_model_name)
            except Exception as e:
                print(f"⚠️ FlashRank initialization warning: {e}")
                self.reranker = None
        else:
            self.reranker = None

    def _ensure_collection_exists(self, vector_size: int = 384) -> bool:
        """Pre-flight check: creates collection with Cosine distance if absent."""
        if not self.client:
            return False
        try:
            from qdrant_client.http import models as qmodels
            existing = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in existing:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE)
                )
            return True
        except Exception:
            return False

    def embed_query(self, query: str) -> List[float]:
        """Embeds query on CPU using FastEmbed."""
        if not self.embedding_model:
            raise RuntimeError("FastEmbed embedding model unavailable.")
        embeddings = list(self.embedding_model.embed([query]))
        emb = embeddings[0]
        return emb.tolist() if hasattr(emb, "tolist") else list(emb)

    def retrieve_candidates(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieves top-20 candidates from Qdrant.
        Gracefully catches 500 errors, timeouts, and network issues.
        """
        if not query or not query.strip():
            return []

        if not self.client:
            return [{"id": "fallback", "text": "Qdrant unavailable; fallback triggered.", "score": 0.0, "metadata": {"error": "no_client"}}]

        try:
            self._ensure_collection_exists()
            query_vector = self.embed_query(query)
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            candidates = []
            for r in search_results:
                payload = getattr(r, "payload", {}) or {}
                text = payload.get("text", payload.get("page_content", ""))
                candidates.append({
                    "id": getattr(r, "id", None),
                    "text": text,
                    "score": getattr(r, "score", 0.0),
                    "metadata": payload
                })
            return candidates
        except Exception as e:
            # Handles Qdrant 500 error / timeout / connection failure
            print(f"⚠️ Qdrant retrieval failure ({type(e).__name__}: {e}) - executing fallback response")
            return [{
                "id": "fallback",
                "text": f"Fallback context: Qdrant service unavailable ({type(e).__name__}).",
                "score": 0.0,
                "metadata": {
                    "fallback": True,
                    "error": f"{type(e).__name__}: {str(e)}",
                    "error_type": type(e).__name__
                }
            }]


    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Reranks candidates to top-5 on CPU using FlashRank.
        """
        if not candidates:
            return []

        # If only fallback exists, return it directly
        if len(candidates) == 1 and candidates[0].get("id") == "fallback":
            return candidates

        if not self.reranker:
            return candidates[:top_k]

        try:
            passages = [
                {"id": c.get("id"), "text": c.get("text", ""), "meta": c.get("metadata", {})}
                for c in candidates
            ]
            rerank_request = RerankRequest(query=query, passages=passages)
            reranked = self.reranker.rerank(rerank_request)
            results = []
            for item in reranked[:top_k]:
                results.append({
                    "id": item.get("id"),
                    "text": item.get("text"),
                    "score": item.get("score"),
                    "metadata": item.get("meta", {})
                })
            return results
        except Exception as e:
            print(f"⚠️ FlashRank reranking error ({type(e).__name__}: {e}) - fallback to raw top_k")
            return candidates[:top_k]

    def retrieve_and_rerank(self, query: str, retrieve_limit: int = 20, rerank_top_k: int = 5) -> List[Dict[str, Any]]:
        """
        End-to-end CPU pipeline:
        1. Embeds query on CPU (FastEmbed, threads=4)
        2. Retrieves top-20 from Qdrant
        3. Reranks to top-5 on CPU (FlashRank)
        """
        candidates = self.retrieve_candidates(query, limit=retrieve_limit)
        return self.rerank(query, candidates, top_k=rerank_top_k)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Context Assembler - Build targeted LLM context from repo graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/context_assembler.py app/main.py
  python scripts/context_assembler.py app/main.py --audit
  python scripts/context_assembler.py app/main.py --depth 3 --budget 120000
  python scripts/context_assembler.py app/main.py --no-source --output overview.md
        """
    )
    parser.add_argument("target", help="Relative path of file to analyze (e.g., app/main.py)")
    parser.add_argument("--audit", action="store_true",
                        help="Enable Audit/QA mode (post-implementation review)")
    parser.add_argument("--depth", type=int, default=DEFAULT_MAX_DEPTH,
                        help=f"Dependency traversal depth (default: {DEFAULT_MAX_DEPTH})")
    parser.add_argument("--budget", type=int, default=DEFAULT_TOKEN_BUDGET,
                        help=f"Token budget (default: {DEFAULT_TOKEN_BUDGET})")
    parser.add_argument("--no-source", action="store_true",
                        help="Only include architecture map, skip source code")
    parser.add_argument("--output", type=str, default="targeted_context.md",
                        help="Output filename (default: targeted_context.md)")

    args = parser.parse_args()

    graph_data = load_graph_data()
    if graph_data:
        build_targeted_context(
            target_file=args.target,
            full_graph_data=graph_data,
            is_audit_mode=args.audit,
            max_depth=args.depth,
            token_budget=args.budget,
            include_source=not args.no_source,
            output_file=args.output
        )