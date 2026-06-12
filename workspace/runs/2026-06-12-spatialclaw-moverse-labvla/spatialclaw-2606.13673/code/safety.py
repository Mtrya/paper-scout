"""Simplified AST sandbox for the probe.

Mirrors SpatialClaw's `spatial_agent/kernel/safety.py` but with a smaller
forbidden list appropriate for a local demonstration.
"""

import ast
import re
from typing import Optional, Set


FORBIDDEN_MODULES: Set[str] = {
    "os", "subprocess", "sys", "shutil", "pathlib", "socket",
    "multiprocessing", "threading", "ctypes", "importlib", "pickle",
    "tempfile", "urllib", "requests",
}

FORBIDDEN_BUILTINS: Set[str] = {"open", "exec", "eval", "compile", "__import__"}

_FILE_IO_PATTERN = re.compile(
    r"\bopen\s*\(|\.read\s*\(|\.write\s*\(|\.save\s*\(|\.to_csv\s*\(",
    re.VERBOSE,
)


def check_code_safety(code: str) -> Optional[str]:
    """Return None if safe, otherwise an error message."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_MODULES:
                    return f"Forbidden import: {alias.name}"
        if isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in FORBIDDEN_MODULES:
                    return f"Forbidden import from: {node.module}"
        if isinstance(node, ast.Call):
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in FORBIDDEN_BUILTINS:
                return f"Forbidden builtin call: {name}()"

    if _FILE_IO_PATTERN.search(code):
        return "Potentially forbidden file I/O detected"
    return None
