"""Minimal persistent Python workspace for the SpatialClaw probe.

This is a humble reconstruction of SpatialClaw's IPython kernel workspace.
Instead of a real Jupyter kernel, it uses a plain Python dict namespace that
persists across "cell" executions, plus stdout capture and a variable tracker.
"""

import ast
import io
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    new_variables: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    execution_time_sec: float = 0.0


class PersistentKernel:
    """A tiny persistent Python execution environment.

    Usage:
        kernel = PersistentKernel()
        kernel.inject("x", 10)
        result = kernel.execute("y = x + 5; print(y)")
        print(kernel.namespace["y"])  # 15
    """

    def __init__(self):
        self.namespace: Dict[str, Any] = {}
        self._previous_vars: set = set()

    def inject(self, name: str, value: Any) -> None:
        """Inject a named object into the namespace."""
        self.namespace[name] = value
        self._previous_vars.add(name)

    def execute(self, code: str, timeout_sec: Optional[float] = None) -> ExecutionResult:
        """Execute a code cell in the persistent namespace.

        Captures stdout/stderr and returns a structured result.  Errors are
        reported as condensed tracebacks, matching SpatialClaw's feedback style.
        """
        import time

        t0 = time.monotonic()
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_buf, stderr_buf

        error = None
        try:
            # Match SpatialClaw: AST-check and execute single interactive block.
            compiled = compile(code, "<agent_cell>", "exec")
            exec(compiled, self.namespace)
        except Exception as exc:
            tb = traceback.format_exc()
            # Condense to the essential line, like SpatialClaw does.
            error = self._condense_traceback(tb)
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

        new_vars = self._diff_namespace()
        elapsed = time.monotonic() - t0
        return ExecutionResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            error=error,
            new_variables=new_vars,
            execution_time_sec=elapsed,
        )

    def _condense_traceback(self, tb: str) -> str:
        """Reduce a traceback to exception type + message + offending line."""
        lines = tb.strip().splitlines()
        if len(lines) < 2:
            return tb
        # Last line is usually "ExceptionType: message"
        exc_line = lines[-1]
        # Find the line inside <agent_cell> that raised.
        for line in reversed(lines):
            if "<agent_cell>" in line:
                return f"{exc_line}\n  -> {line.strip()}"
        return exc_line

    def _diff_namespace(self) -> Dict[str, Dict[str, Any]]:
        """Return variables created or changed since last call."""
        current = set(self.namespace.keys())
        new = current - self._previous_vars
        self._previous_vars = current
        info: Dict[str, Dict[str, Any]] = {}
        for name in new:
            val = self.namespace[name]
            if name.startswith("_"):
                continue
            if callable(val):
                continue
            entry: Dict[str, Any] = {"type": type(val).__name__}
            if hasattr(val, "shape"):
                entry["shape"] = str(val.shape)
            if hasattr(val, "dtype"):
                entry["dtype"] = str(val.dtype)
            if isinstance(val, (list, tuple, dict)):
                entry["len"] = len(val)
            info[name] = entry
        return info

    def get_variables(self) -> Dict[str, Dict[str, Any]]:
        """Return a snapshot of non-internal variables."""
        info: Dict[str, Any] = {}
        for name, val in self.namespace.items():
            if name.startswith("_") or callable(val):
                continue
            entry = {"type": type(val).__name__}
            if hasattr(val, "shape"):
                entry["shape"] = str(val.shape)
            if hasattr(val, "dtype"):
                entry["dtype"] = str(val.dtype)
            if isinstance(val, (list, tuple, dict)):
                entry["len"] = len(val)
            info[name] = entry
        return info
