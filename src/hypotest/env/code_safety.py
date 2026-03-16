"""Code safety checker for model-generated Python/R code.

Prevents model-generated code from killing processes, destroying filesystems,
or escaping the execution sandbox. Uses AST analysis for Python (with regex
fallback) and regex for R code.
"""

from __future__ import annotations

import ast
import logging
import re

from .kernel_server import NBLanguage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Category messages — intentionally vague to avoid teaching evasion
# ---------------------------------------------------------------------------
_MSG_BLOCKED_IMPORT = "Code blocked: imports a restricted module."
_MSG_DANGEROUS_CALL = "Code blocked: calls a restricted function."
_MSG_BLOCKED_BUILTIN = "Code blocked: uses a restricted builtin."
_MSG_SHELL_ESCAPE = "Code blocked: shell escape is not allowed."
_MSG_DANGEROUS_SHELL = "Code blocked: contains a restricted shell command."
_MSG_R_RESTRICTED = "Code blocked: calls a restricted R function."


# ---------------------------------------------------------------------------
# Blocked modules (importing these is never allowed)
# ---------------------------------------------------------------------------
_BLOCKED_MODULES: frozenset[str] = frozenset({
    "ctypes",
    "signal",
})

# ---------------------------------------------------------------------------
# Dangerous module.function pairs
# ---------------------------------------------------------------------------
_DANGEROUS_CALLS: frozenset[str] = frozenset({
    # os — process management
    "os.kill",
    "os.killpg",
    "os.system",
    "os.popen",
    "os.fork",
    "os.forkpty",
    "os.execl",
    "os.execle",
    "os.execlp",
    "os.execlpe",
    "os.execv",
    "os.execve",
    "os.execvp",
    "os.execvpe",
    "os._exit",
    # subprocess
    "subprocess.run",
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    # shutil — destructive
    "shutil.rmtree",
    # multiprocessing — process spawning
    "multiprocessing.Process",
})

# Dangerous attribute names used with getattr() evasion
_DANGEROUS_ATTRS: frozenset[str] = frozenset({
    "kill",
    "killpg",
    "system",
    "popen",
    "fork",
    "forkpty",
    "execl",
    "execle",
    "execlp",
    "execlpe",
    "execv",
    "execve",
    "execvp",
    "execvpe",
    "_exit",
    "rmtree",
})

# Modules whose import_module() calls should be checked
_DANGEROUS_IMPORTLIB_TARGETS: frozenset[str] = _BLOCKED_MODULES | frozenset({
    "subprocess",
    "shutil",
    "multiprocessing",
})


# ---------------------------------------------------------------------------
# AST Walker for Python
# ---------------------------------------------------------------------------
class _DangerousCodeVisitor(ast.NodeVisitor):
    """Walks Python AST to detect dangerous calls, tracking import aliases."""

    def __init__(self) -> None:
        self.aliases: dict[str, str] = {}  # alias -> canonical module path
        self.blocked_reason: str | None = None

    def _block(self, reason: str) -> None:
        if self.blocked_reason is None:
            self.blocked_reason = reason

    # -- Import tracking -----------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            module = alias.name
            local_name = alias.asname or alias.name
            # Check top-level module against blocked list
            top_module = module.split(".")[0]
            if top_module in _BLOCKED_MODULES:
                self._block(_MSG_BLOCKED_IMPORT)
                return
            self.aliases[local_name] = module
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        top_module = module.split(".")[0]
        if top_module in _BLOCKED_MODULES:
            self._block(_MSG_BLOCKED_IMPORT)
            return
        for alias in node.names:
            local_name = alias.asname or alias.name
            self.aliases[local_name] = f"{module}.{alias.name}"
        self.generic_visit(node)

    # -- Call detection ------------------------------------------------------

    def _resolve_call_path(self, node: ast.Call) -> str | None:
        """Resolve a Call node to a canonical 'module.func' string."""
        func = node.func
        if isinstance(func, ast.Attribute):
            # e.g. os.kill(...) or sp.run(...)
            parts: list[str] = [func.attr]
            obj = func.value
            while isinstance(obj, ast.Attribute):
                parts.append(obj.attr)
                obj = obj.value
            if isinstance(obj, ast.Name):
                parts.append(obj.id)
                parts.reverse()
                # Resolve the leftmost name via aliases
                base = parts[0]
                if base in self.aliases:
                    canonical_base = self.aliases[base]
                    return f"{canonical_base}.{'.'.join(parts[1:])}"
                return ".".join(parts)
        elif isinstance(func, ast.Name):
            # e.g. kill(...) — might be from `from os import kill`
            if func.id in self.aliases:
                return self.aliases[func.id]
            return func.id
        return None

    def visit_Call(self, node: ast.Call) -> None:  # noqa: PLR0912
        call_path = self._resolve_call_path(node)

        if call_path is not None:
            # Check exec() builtin
            if call_path == "exec" or call_path.endswith(".exec"):
                # Only block the builtin exec, not method calls like cursor.exec
                func = node.func
                if isinstance(func, ast.Name) and func.id == "exec":
                    self._block(_MSG_BLOCKED_BUILTIN)
                    self.generic_visit(node)
                    return

            # Check against dangerous calls
            for dangerous in _DANGEROUS_CALLS:
                if call_path == dangerous or call_path.endswith(f".{dangerous}"):
                    self._block(_MSG_DANGEROUS_CALL)
                    self.generic_visit(node)
                    return

            # Check __import__('os') style
            if call_path == "__import__" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    top = arg.value.split(".")[0]
                    if top in _DANGEROUS_IMPORTLIB_TARGETS or top in _BLOCKED_MODULES:
                        self._block(_MSG_DANGEROUS_CALL)
                        self.generic_visit(node)
                        return

            # Check getattr(os, 'kill') style
            if call_path in {"getattr", "builtins.getattr"} and len(node.args) >= 2:
                attr_arg = node.args[1]
                if (
                    isinstance(attr_arg, ast.Constant)
                    and isinstance(attr_arg.value, str)
                    and attr_arg.value in _DANGEROUS_ATTRS
                ):
                    self._block(_MSG_DANGEROUS_CALL)
                    self.generic_visit(node)
                    return

            # Check importlib.import_module('signal') style
            if call_path.endswith("import_module") and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    top = arg.value.split(".")[0]
                    if top in _DANGEROUS_IMPORTLIB_TARGETS or top in _BLOCKED_MODULES:
                        self._block(_MSG_BLOCKED_IMPORT)
                        self.generic_visit(node)
                        return

        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Jupyter Magic Patterns (Python pre-pass)
# ---------------------------------------------------------------------------
# Shell escape line: !cmd
_MAGIC_SHELL_ESCAPE_LINE = re.compile(r"^\s*!(?P<cmd>.+)$", re.MULTILINE)

# Dangerous IPython magics
_MAGIC_DANGEROUS = re.compile(r"^\s*%(run|system)\b|get_ipython\(\)\s*\.\s*system\s*\(", re.MULTILINE)


def _check_jupyter_magics(code: str) -> str | None:
    """Check for dangerous Jupyter magics. Returns reason or None."""
    # Check dangerous magics first (always blocked)
    if _MAGIC_DANGEROUS.search(code):
        return _MSG_SHELL_ESCAPE

    # For !cmd shell escapes, extract the command and check against _BASH_PATTERNS.
    # Everything is allowed unless the command matches a dangerous bash pattern.
    for match in _MAGIC_SHELL_ESCAPE_LINE.finditer(code):
        cmd_text = match.group("cmd").strip()
        if _check_bash_patterns(cmd_text) is not None:
            return _MSG_DANGEROUS_SHELL

    return None


# ---------------------------------------------------------------------------
# Bash-level patterns (both languages)
# ---------------------------------------------------------------------------
_BASH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bkillall\b"),
    re.compile(r"\bpkill\b"),
    re.compile(r"\bkill\s+.*\$\("),
    re.compile(r"\bkill\s+.*`"),
    re.compile(r"\bkill\s+(-\d+\s+|-[A-Z]+\s+|-SIG[A-Z]+\s+)*\$\w+"),
    re.compile(r"\bkill\s+(-\d+\s+|-[A-Z]+\s+|-SIG[A-Z]+\s+)*-1\b"),
    re.compile(r"\bkill\s+(-\d+\s+|-[A-Z]+\s+|-SIG[A-Z]+\s+)*0\b"),
    re.compile(r"\b(shutdown|reboot|poweroff|halt|init\s+[06])\b"),
    re.compile(r"\bdd\s+.*of=\s*/dev/"),
    re.compile(r"\brm\s+(-\w+\s+)*(/\s*$|/\*)"),
    re.compile(r"\brm\s+(-\w+\s+)*(/(bin|usr|etc|var|home|root|opt|lib|lib64|sbin|boot|dev|proc|sys))\b"),
]


def _check_bash_patterns(code: str) -> str | None:
    """Check for dangerous bash-level patterns. Returns reason or None."""
    for pattern in _BASH_PATTERNS:
        if pattern.search(code):
            return _MSG_DANGEROUS_SHELL
    return None


# ---------------------------------------------------------------------------
# R regex patterns
# ---------------------------------------------------------------------------
_R_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bsystem\s*\("),
    re.compile(r"\bsystem2\s*\("),
    re.compile(r"\bshell\s*\("),
    re.compile(r"\bshell\.exec\s*\("),
    re.compile(r"\bproc\.kill\b"),
    re.compile(r"\bquit\s*\("),
    re.compile(r"\bq\s*\(\s*\)"),
]


def _check_r_patterns(code: str) -> str | None:
    """Check R code for dangerous patterns. Returns reason or None."""
    for pattern in _R_PATTERNS:
        if pattern.search(code):
            return _MSG_R_RESTRICTED
    return None


# ---------------------------------------------------------------------------
# Python regex fallback (when AST parse fails)
# ---------------------------------------------------------------------------
_PYTHON_REGEX_FALLBACK: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bos\s*\.\s*kill\s*\("), _MSG_DANGEROUS_CALL),
    (re.compile(r"\bos\s*\.\s*killpg\s*\("), _MSG_DANGEROUS_CALL),
    (re.compile(r"\bos\s*\.\s*system\s*\("), _MSG_DANGEROUS_CALL),
    (re.compile(r"\bos\s*\.\s*popen\s*\("), _MSG_DANGEROUS_CALL),
    (re.compile(r"\bos\s*\.\s*fork\s*\("), _MSG_DANGEROUS_CALL),
    (re.compile(r"\bos\s*\.\s*exec\w*\s*\("), _MSG_DANGEROUS_CALL),
    (re.compile(r"\bos\s*\.\s*_exit\s*\("), _MSG_DANGEROUS_CALL),
    (re.compile(r"\bsubprocess\s*\.\s*(run|Popen|call|check_call|check_output)\s*\("), _MSG_DANGEROUS_CALL),
    (re.compile(r"\bshutil\s*\.\s*rmtree\s*\("), _MSG_DANGEROUS_CALL),
    (re.compile(r"\bmultiprocessing\s*\.\s*Process\s*\("), _MSG_DANGEROUS_CALL),
    (re.compile(r"\b__import__\s*\("), _MSG_DANGEROUS_CALL),
    (re.compile(r"\bexec\s*\("), _MSG_BLOCKED_BUILTIN),
    (re.compile(r"\bimport\s+ctypes\b"), _MSG_BLOCKED_IMPORT),
    (re.compile(r"\bimport\s+signal\b"), _MSG_BLOCKED_IMPORT),
    (re.compile(r"\bfrom\s+ctypes\b"), _MSG_BLOCKED_IMPORT),
    (re.compile(r"\bfrom\s+signal\b"), _MSG_BLOCKED_IMPORT),
]


def _check_python_regex_fallback(code: str) -> str | None:
    """Regex fallback for Python when AST parsing fails."""
    for pattern, msg in _PYTHON_REGEX_FALLBACK:
        if pattern.search(code):
            return msg
    return None


# ---------------------------------------------------------------------------
# Python AST check with fallback
# ---------------------------------------------------------------------------
def _check_python(code: str) -> str | None:
    """Three-layer Python check: magic pre-pass, AST, regex fallback."""
    # Layer 1: Jupyter magic pre-pass (not valid Python AST)
    magic_result = _check_jupyter_magics(code)
    if magic_result is not None:
        return magic_result

    # Layer 2: AST analysis
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Try stripping magic lines and re-parsing
        stripped_lines = []
        for line in code.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("!", "%")):
                stripped_lines.append("")  # blank out magic lines
            else:
                stripped_lines.append(line)
        stripped_code = "\n".join(stripped_lines)
        try:
            tree = ast.parse(stripped_code)
        except SyntaxError:
            # Layer 3: Regex fallback
            return _check_python_regex_fallback(code)

    visitor = _DangerousCodeVisitor()
    visitor.visit(tree)
    return visitor.blocked_reason


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def check_code_safety(code: str, language: NBLanguage) -> str | None:
    """Check if code is safe to execute.

    Args:
        code: The code string to check.
        language: The notebook language (PYTHON or R).

    Returns:
        None if the code is safe, or a category-level error message if blocked.
    """
    # Bash-level patterns apply to both languages
    bash_result = _check_bash_patterns(code)
    if bash_result is not None:
        code_snippet = code[:200] + ("..." if len(code) > 200 else "")
        logger.warning("Code safety block: reason=%r language=%s code=%r", bash_result, language.value, code_snippet)
        return bash_result

    result: str | None = None
    if language == NBLanguage.PYTHON:
        result = _check_python(code)
    elif language == NBLanguage.R:
        result = _check_r_patterns(code)

    if result is not None:
        code_snippet = code[:200] + ("..." if len(code) > 200 else "")
        logger.warning("Code safety block: reason=%r language=%s code=%r", result, language.value, code_snippet)

    return result
