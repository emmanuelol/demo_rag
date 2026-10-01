#!/usr/bin/env python3
"""
Repository Mapper - Builds a comprehensive code graph (JSON) of the entire repository.
Captures structure, dependencies, docstrings, complexity, and call relationships.
"""

import os
import ast
import json
import fnmatch
import subprocess
from pathlib import Path
from typing import Optional
from collections import defaultdict

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# === CONFIGURATION ===
EXCLUDE_DIRS = {
    '.git', '.pytest_cache', '__pycache__', 'venv', 'env', '.venv', '.venv_app',
    'node_modules', 'dist', 'build', '.deps', 'rollbacks', 'scratch',
    '.vscode', '.ruff_cache', 'ruff_cache', '.checkpoints', '.codegraph',
    '.obsidian', '_jobs', '_agents', 'archive', 'knowledge', '.mypy_cache',
    '.tox', 'site-packages', 'qa_images', 'shared_media', 'openclaw_data',
    '.bin', '.dumbledoer', 'checkpoint'
}

EXCLUDE_EXTENSIONS = {'.pyc', '.pyo', '.so', '.dylib', '.whl', '.egg'}

REPO_ROOT = Path(__file__).resolve().parent.parent
INCLUDE_TESTS = False
MAX_FILE_SIZE = 512 * 1024  # 512KB limit per file


# ============================================================
# AST ANALYZER - Enhanced with docstrings, signatures, calls
# ============================================================

class CodeGraphBuilder(ast.NodeVisitor):
    """Extracts rich structural information from Python AST."""

    def __init__(self, filepath: str, source_lines: list[str]):
        self.filepath = filepath
        self.source_lines = source_lines
        self.classes = []
        self.functions = []
        self.imports = []
        self.import_froms = []
        self.global_calls = []
        self.module_docstring = ""
        self._current_class = None
        self._function_depth = 0

    def visit_Module(self, node: ast.Module):
        self.module_docstring = ast.get_docstring(node) or ""
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append({
                "module": alias.name,
                "alias": alias.asname,
                "line": node.lineno
            })
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        level = node.level  # For relative imports
        names = [{"name": a.name, "alias": a.asname} for a in node.names]
        self.import_froms.append({
            "module": module,
            "level": level,
            "names": names,
            "line": node.lineno
        })
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_info = self._extract_function_info(item)
                method_info["is_method"] = True
                methods.append(method_info)

        decorators = [self._get_decorator_name(d) for d in node.decorator_list]
        bases = [self._get_node_name(b) for b in node.bases]

        class_info = {
            "name": node.name,
            "line": node.lineno,
            "end_line": getattr(node, 'end_lineno', node.lineno),
            "docstring": ast.get_docstring(node) or "",
            "decorators": decorators,
            "bases": bases,
            "methods": methods
        }

        if self._current_class is None and self._function_depth == 0:
            self.classes.append(class_info)

        old_class = self._current_class
        self._current_class = node.name
        self.generic_visit(node)
        self._current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self._current_class is None and self._function_depth == 0:
            func_info = self._extract_function_info(node)
            func_info["is_method"] = False
            self.functions.append(func_info)

        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        if self._current_class is None and self._function_depth == 0:
            func_info = self._extract_function_info(node)
            func_info["is_method"] = False
            func_info["is_async"] = True
            self.functions.append(func_info)

        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1

    def visit_Call(self, node: ast.Call):
        """Track top-level calls when outside class/function definitions."""
        if self._current_class is None and self._function_depth == 0:
            call_name = self._get_node_name(node.func)
            if call_name:
                self.global_calls.append({
                    "callee": call_name,
                    "line": node.lineno
                })
        self.generic_visit(node)

    def _extract_function_info(self, node) -> dict:
        """Extract comprehensive function metadata."""
        args = self._extract_args(node.args)
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]

        internal_calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_name = self._get_node_name(child.func)
                if call_name:
                    internal_calls.append(call_name)

        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor,
                                  ast.ExceptHandler, ast.With, ast.AsyncWith,
                                  ast.BoolOp)):
                complexity += 1
            elif isinstance(child, ast.comprehension):
                complexity += 1

        return {
            "name": node.name,
            "line": node.lineno,
            "end_line": getattr(node, 'end_lineno', node.lineno),
            "docstring": ast.get_docstring(node) or "",
            "args": args,
            "return_annotation": self._get_annotation_str(node.returns),
            "decorators": decorators,
            "internal_calls": sorted(list(set(internal_calls))),
            "complexity": complexity,
            "is_async": isinstance(node, ast.AsyncFunctionDef)
        }

    def _extract_args(self, args_node) -> list[dict]:
        """Extract argument names, types, and defaults."""
        result = []
        defaults_offset = len(args_node.args) - len(args_node.defaults)

        for i, arg in enumerate(args_node.args):
            arg_info = {
                "name": arg.arg,
                "annotation": self._get_annotation_str(arg.annotation),
            }
            default_idx = i - defaults_offset
            if 0 <= default_idx < len(args_node.defaults):
                arg_info["has_default"] = True
            result.append(arg_info)

        for arg in args_node.kwonlyargs:
            result.append({
                "name": arg.arg,
                "annotation": self._get_annotation_str(arg.annotation),
                "keyword_only": True
            })

        if args_node.vararg:
            result.append({"name": f"*{args_node.vararg.arg}", "vararg": True})
        if args_node.kwarg:
            result.append({"name": f"**{args_node.kwarg.arg}", "kwarg": True})

        return result

    def _get_annotation_str(self, node) -> Optional[str]:
        """Convert annotation AST node to string."""
        if node is None:
            return None
        try:
            return ast.unparse(node)
        except Exception:
            return str(node)

    def _get_decorator_name(self, node) -> str:
        """Extract decorator name."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            val = self._get_node_name(node.value)
            return f"{val}.{node.attr}" if val else node.attr
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        return "<complex>"

    def _get_node_name(self, node) -> str:
        """Get a string representation of a node (for calls, bases, etc.)."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_node_name(node.value)
            return f"{value}.{node.attr}" if value else node.attr
        elif isinstance(node, ast.Call):
            return self._get_node_name(node.func)
        elif isinstance(node, ast.Subscript):
            return self._get_node_name(node.value)
        return ""


def analyze_python_file(filepath: Path) -> dict:
    """Parse a Python file and extract rich structural metadata."""
    try:
        source = filepath.read_text(encoding='utf-8', errors='replace')
        source_lines = source.splitlines()
        tree = ast.parse(source, filename=str(filepath))

        builder = CodeGraphBuilder(str(filepath), source_lines)
        builder.visit(tree)

        total_functions = len(builder.functions) + sum(len(c["methods"]) for c in builder.classes)
        total_complexity = sum(f["complexity"] for f in builder.functions)
        total_complexity += sum(m["complexity"] for c in builder.classes for m in c["methods"])

        return {
            "type": "python",
            "module_docstring": builder.module_docstring,
            "imports": builder.imports,
            "import_froms": builder.import_froms,
            "classes": builder.classes,
            "functions": builder.functions,
            "top_level_calls": builder.global_calls[:50],
            "metrics": {
                "total_functions": total_functions,
                "total_classes": len(builder.classes),
                "total_complexity": total_complexity,
                "line_count": len(source_lines),
            },
            "summary": _generate_file_summary(builder)
        }

    except SyntaxError as e:
        return {"type": "python", "error": f"SyntaxError: {e}", "line": e.lineno}
    except Exception as e:
        return {"type": "python", "error": str(e)}


def _generate_file_summary(builder: CodeGraphBuilder) -> str:
    """Auto-generate a human-readable summary of what this file does."""
    parts = []

    if builder.module_docstring:
        first_line = builder.module_docstring.strip().splitlines()[0]
        if first_line:
            parts.append(first_line[:200])

    if builder.classes:
        class_names = [c["name"] for c in builder.classes]
        parts.append(f"Defines classes: {', '.join(class_names)}")

    if builder.functions:
        func_names = [f["name"] for f in builder.functions if not f["name"].startswith("_")]
        if func_names:
            parts.append(f"Exposes functions: {', '.join(func_names[:10])}")

    if builder.import_froms:
        modules = sorted(list(set(imp["module"] for imp in builder.import_froms if imp["module"])))
        if modules:
            parts.append(f"Depends on: {', '.join(modules[:8])}")

    return " | ".join(parts) if parts else "No summary available."


# ============================================================
# NON-PYTHON ANALYZERS
# ============================================================

def analyze_dockerfile(filepath: Path) -> dict:
    """Parse Dockerfile for base images, ports, entrypoint."""
    data = {"type": "docker", "base_images": [], "exposed_ports": [],
            "entrypoint": None, "cmd": None, "stages": []}
    try:
        for line in filepath.read_text(encoding='utf-8', errors='replace').splitlines():
            line = line.strip()
            if line.startswith("FROM "):
                img = line[5:].strip()
                data["base_images"].append(img)
                data["stages"].append(img)
            elif line.startswith("EXPOSE "):
                data["exposed_ports"].append(line[7:].strip())
            elif line.startswith("ENTRYPOINT "):
                data["entrypoint"] = line[11:].strip()
            elif line.startswith("CMD "):
                data["cmd"] = line[4:].strip()
        data["summary"] = f"Docker image based on {data['base_images'][0] if data['base_images'] else 'unknown'}"
    except Exception as e:
        data["error"] = str(e)
    return data


def analyze_config_file(filepath: Path) -> dict:
    """Parse JSON/YAML config files."""
    ext = filepath.suffix.lower()
    data = {"type": "config_file", "format": ext, "top_level_keys": []}
    try:
        content = filepath.read_text(encoding='utf-8', errors='replace')
        if ext == '.json':
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                data["top_level_keys"] = list(parsed.keys())
                data["summary"] = f"JSON config with keys: {', '.join(list(parsed.keys())[:10])}"
        elif ext in ['.yml', '.yaml'] and HAS_YAML:
            parsed = yaml.safe_load(content)
            if isinstance(parsed, dict):
                data["top_level_keys"] = list(parsed.keys())
                data["summary"] = f"YAML config with keys: {', '.join(list(parsed.keys())[:10])}"
        else:
            data["summary"] = f"Config file ({ext})"
    except Exception as e:
        data["error"] = str(e)
    return data


def analyze_generic_file(filepath: Path) -> dict:
    """Basic info for text/markdown/shell files."""
    try:
        content = filepath.read_text(encoding='utf-8', errors='replace')
        lines = content.splitlines()
        summary = ""
        for line in lines[:20]:
            line = line.strip()
            if line and not line.startswith('#!') and not line.startswith('---'):
                summary = line[:150]
                break
        return {
            "type": "generic_text",
            "size_bytes": filepath.stat().st_size,
            "line_count": len(lines),
            "summary": summary
        }
    except Exception:
        return {"type": "generic", "error": "Unreadable"}


# ============================================================
# GIT UTILITIES
# ============================================================

def get_git_branch(repo_path: Path) -> str:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=repo_path, capture_output=True, text=True, check=True, timeout=10
        )
        return result.stdout.strip()
    except Exception:
        return "unknown-branch"


def get_git_modified_files(repo_path: Path) -> set:
    """Get modified and untracked files for incremental mode."""
    modified = set()
    try:
        diff_res = subprocess.run(
            ['git', 'diff', '--name-only', 'HEAD'],
            cwd=repo_path, capture_output=True, text=True, check=False, timeout=10
        )
        if diff_res.returncode == 0:
            modified.update(diff_res.stdout.strip().splitlines())

        status_res = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=repo_path, capture_output=True, text=True, check=False, timeout=10
        )
        if status_res.returncode == 0:
            for line in status_res.stdout.splitlines():
                if len(line) > 3:
                    file_part = line[3:].strip()
                    if ' -> ' in file_part:
                        file_part = file_part.split(' -> ')[1].strip()
                    modified.add(file_part)
    except Exception:
        pass
    return {f for f in modified if f}


def load_gitignore_patterns(root: Path) -> list:
    """Load .gitignore patterns."""
    patterns = []
    gitignore_path = root / '.gitignore'
    if gitignore_path.exists():
        for line in gitignore_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('!'):
                if line.endswith('/'):
                    line = line[:-1]
                patterns.append(line)
    return patterns


def is_ignored_dir(dir_name: str, rel_dir: str, gitignore_patterns: list) -> bool:
    """Check if directory should be skipped during walk."""
    if dir_name.startswith('.') and dir_name != '.':
        return True
    if dir_name in EXCLUDE_DIRS:
        return True
    if not INCLUDE_TESTS and dir_name in ('tests', 'test'):
        return True
    for pattern in gitignore_patterns:
        if fnmatch.fnmatch(rel_dir, pattern) or fnmatch.fnmatch(dir_name, pattern):
            return True
    return False


def is_ignored_file(path: Path, gitignore_patterns: list, repo_root: Path) -> bool:
    """Check if a file should be excluded from mapping."""
    try:
        rel_path_obj = path.relative_to(repo_root)
    except ValueError:
        return True

    rel_path = str(rel_path_obj)

    if path.suffix.lower() in EXCLUDE_EXTENSIONS:
        return True

    for pattern in gitignore_patterns:
        if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(path.name, pattern):
            return True

    try:
        if path.stat().st_size > MAX_FILE_SIZE:
            return True
    except OSError:
        return True

    return False


# ============================================================
# GRAPH BUILDER
# ============================================================

def build_dependency_edges(graph: dict) -> list[dict]:
    """Build explicit dependency edges between files with deterministic matching."""
    edges_set = set()

    for source_file, data in graph.items():
        if data.get("type") != "python":
            continue

        for imp in data.get("imports", []):
            module = imp.get("module", "") if isinstance(imp, dict) else str(imp)
            target = _resolve_import_to_file(module, graph, source_file)
            if target and target != source_file:
                edges_set.add((source_file, target, "imports"))

        for imp in data.get("import_froms", []):
            module = imp.get("module", "")
            level = imp.get("level", 0)
            names = imp.get("names", [])

            target = _resolve_from_import_to_file(module, level, names, graph, source_file)
            if target and target != source_file:
                edges_set.add((source_file, target, "from_import"))

    return [
        {"from": s, "to": t, "type": ty}
        for s, t, ty in sorted(edges_set, key=lambda x: (x[0], x[1], x[2]))
    ]


def _resolve_import_to_file(module_name: str, graph: dict, source_file: Optional[str] = None) -> Optional[str]:
    """Resolve an exact module import to a file path in graph."""
    if not module_name:
        return None

    parts = module_name.split('.')
    path_as_file = '/'.join(parts) + '.py'
    if path_as_file in graph:
        return path_as_file

    path_as_init = '/'.join(parts) + '/__init__.py'
    if path_as_init in graph:
        return path_as_init

    if source_file and source_file.startswith('app/'):
        app_file = 'app/' + path_as_file
        if app_file in graph:
            return app_file
        app_init = 'app/' + path_as_init
        if app_init in graph:
            return app_init

    return None


def _resolve_from_import_to_file(
    module_name: str, level: int, names: list[dict], graph: dict, source_file: str
) -> Optional[str]:
    """Resolve a 'from X import Y' statement accurately without substring collision."""
    source_dir = Path(source_file).parent

    if level > 0:
        base_dir = source_dir
        for _ in range(level - 1):
            base_dir = base_dir.parent

        if module_name:
            rel_path = (base_dir / '/'.join(module_name.split('.'))).as_posix()
        else:
            rel_path = base_dir.as_posix()

        if rel_path == '.':
            rel_path = ''

        if rel_path and f"{rel_path}.py" in graph:
            return f"{rel_path}.py"
        if rel_path and f"{rel_path}/__init__.py" in graph:
            return f"{rel_path}/__init__.py"
        for name_info in names:
            sym_name = name_info.get("name", "")
            target_candidate = f"{rel_path}/{sym_name}.py" if rel_path else f"{sym_name}.py"
            if target_candidate in graph:
                return target_candidate

    if module_name:
        direct = _resolve_import_to_file(module_name, graph, source_file)
        if direct:
            return direct

        for name_info in names:
            sym_name = name_info.get("name", "")
            submodule = f"{module_name}.{sym_name}"
            sub_direct = _resolve_import_to_file(submodule, graph, source_file)
            if sub_direct:
                return sub_direct

    return None


def compute_node_importance(graph: dict, edges: list) -> dict:
    """Compute importance score based on connectivity."""
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)

    for edge in edges:
        out_degree[edge["from"]] += 1
        in_degree[edge["to"]] += 1

    importance = {}
    for file_path in graph:
        score = in_degree.get(file_path, 0) * 2 + out_degree.get(file_path, 0)
        importance[file_path] = score

    return importance


def build_repository_graph(root_dir: Path = REPO_ROOT, incremental: bool = False):
    """Main entry point: scan repo and build the complete graph."""
    root_path = Path(root_dir)
    gitignore_patterns = load_gitignore_patterns(root_path)
    current_branch = get_git_branch(root_path)

    print(f"{'='*60}")
    print(f"  Repository Mapper")
    print(f"  Root: {root_path}")
    print(f"  Branch: {current_branch}")
    print(f"  Mode: {'Incremental' if incremental else 'Full scan'}")
    print(f"{'='*60}")

    existing_graph = {}
    output_file = root_path / "local_codegraph.json"
    modified_files = set()
    if incremental and output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                old_data = json.load(f)
                existing_graph = old_data.get("files", {})
            modified_files = get_git_modified_files(root_path)
            print(f"  Modified files detected: {len(modified_files)}")
        except Exception as e:
            print(f"  Incremental load failed ({e}), falling back to full scan.")
            incremental = False

    graph = {}
    mapped_count = 0
    skipped_count = 0
    error_count = 0

    for dirpath, dirnames, filenames in os.walk(root_path):
        rel_dir = os.path.relpath(dirpath, root_path)
        if rel_dir == '.':
            rel_dir = ''

        # Prune excluded directories in-place
        dirnames[:] = [
            d for d in sorted(dirnames)
            if not is_ignored_dir(d, os.path.join(rel_dir, d) if rel_dir else d, gitignore_patterns)
        ]

        for filename in sorted(filenames):
            filepath = Path(dirpath) / filename
            if is_ignored_file(filepath, gitignore_patterns, root_path):
                skipped_count += 1
                continue

            rel_path = str(filepath.relative_to(root_path))

            if incremental and rel_path not in modified_files and rel_path in existing_graph:
                graph[rel_path] = existing_graph[rel_path]
                mapped_count += 1
                continue

            ext = filepath.suffix.lower()

            file_data = None
            if ext == '.py':
                file_data = analyze_python_file(filepath)
            elif filename == 'Dockerfile' or filename.startswith('Dockerfile.') or filename.endswith('.dockerfile'):
                file_data = analyze_dockerfile(filepath)
            elif ext in ['.json', '.yml', '.yaml']:
                file_data = analyze_config_file(filepath)
            elif ext in ['.md', '.txt', '.sh', '.toml', '.cfg', '.ini']:
                file_data = analyze_generic_file(filepath)
            else:
                continue

            if file_data:
                if file_data.get("error"):
                    error_count += 1
                graph[rel_path] = file_data
                mapped_count += 1

    print(f"\n  Building dependency edges...")
    edges = build_dependency_edges(graph)
    importance = compute_node_importance(graph, edges)

    output_data = {
        "metadata": {
            "branch": current_branch,
            "mapped_files_count": mapped_count,
            "skipped_files_count": skipped_count,
            "error_files_count": error_count,
            "total_edges": len(edges),
            "generated_by": "map_repository.py v2.0"
        },
        "files": graph,
        "edges": edges,
        "importance_scores": importance
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  ✓ Graph built successfully!")
    print(f"  Files mapped: {mapped_count}")
    print(f"  Edges found:  {len(edges)}")
    print(f"  Errors:       {error_count}")
    print(f"  Output:       {output_file}")
    print(f"{'='*60}")

    return output_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Map repository into a code graph")
    parser.add_argument("--incremental", action="store_true",
                        help="Only re-parse files changed since last commit")
    parser.add_argument("--root", type=str, default=None,
                        help="Override repository root path")
    args = parser.parse_args()

    root = Path(args.root) if args.root else REPO_ROOT
    build_repository_graph(root_dir=root, incremental=args.incremental)