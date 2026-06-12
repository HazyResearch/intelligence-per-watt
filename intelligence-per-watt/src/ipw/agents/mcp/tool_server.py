"""Tool servers for non-LM utilities (search, code execution, calculator)."""

from __future__ import annotations

import ast
import math
import operator
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from typing import Any, List, Optional

from ipw.agents.mcp.base import BaseMCPServer, MCPToolResult


def _resolve_workspace_abbrev_path(file_path: str, working_dir: str | None) -> str:
    """Map model-copied `/.../workspace/...` paths back into the active workspace."""
    if not working_dir or "/.../workspace/" not in file_path:
        return file_path
    suffix = file_path.split("/.../workspace/", 1)[1]
    return os.path.join(str(working_dir), suffix)


def _check_bubblewrap_available() -> bool:
    """Check if bubblewrap (bwrap) is available on the system."""
    return shutil.which("bwrap") is not None


def _get_python_lib_paths() -> List[str]:
    """Get Python library paths that need to be mounted read-only."""
    paths = set()

    # Add sys.path entries that exist
    for path in sys.path:
        if path and os.path.exists(path):
            # Get the real path (resolve symlinks)
            real_path = os.path.realpath(path)
            paths.add(real_path)

    # Add common system paths
    system_paths = [
        "/usr",
        "/lib",
        "/lib64",
        "/etc/alternatives",
        "/etc/ld.so.cache",
        "/etc/ld.so.conf",
        "/etc/ld.so.conf.d",
    ]
    for path in system_paths:
        if os.path.exists(path):
            paths.add(path)

    # Add Python executable's directory
    python_dir = os.path.dirname(os.path.realpath(sys.executable))
    if os.path.exists(python_dir):
        paths.add(python_dir)

    return sorted(paths)


class CalculatorServer(BaseMCPServer):
    """MCP server for mathematical calculations.

    Safely evaluates mathematical expressions using AST.

    Example:
        calc = CalculatorServer()
        result = calc.execute("2 + 2 * 3")
        print(result.content)  # "8"
    """

    # Safe operators for math evaluation
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    FUNCTIONS = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "exp": math.exp,
        "abs": abs,
        "round": round,
    }

    def __init__(self, telemetry_collector: Optional[Any] = None):
        super().__init__(
            name="calculator",
            telemetry_collector=telemetry_collector,
        )

    def _execute_impl(self, prompt: str, **params: Any) -> MCPToolResult:
        """Execute mathematical calculation.

        Args:
            prompt: Mathematical expression to evaluate

        Returns:
            MCPToolResult with calculated result
        """
        # Extract expression from prompt
        expression = self._extract_expression(prompt)

        try:
            result = self._safe_eval(expression)
            content = str(result)
            error = None
        except Exception as e:
            content = f"Error: {e}"
            error = str(e)

        return MCPToolResult(
            content=content,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            cost_usd=0.0,
            metadata={
                "tool": "calculator",
                "expression": expression,
                "error": error,
            },
        )

    def _extract_expression(self, prompt: str) -> str:
        """Extract mathematical expression from prompt."""
        # Look for expression in common formats
        patterns = [
            r"calculate\s+(.+)",
            r"compute\s+(.+)",
            r"evaluate\s+(.+)",
            r"what\s+is\s+(.+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, prompt.lower())
            if match:
                expr = match.group(1).strip()
                # Strip trailing punctuation
                expr = expr.rstrip("?!.,;")
                return expr

        # If no pattern matches, assume entire prompt is the expression
        return prompt.strip().rstrip("?!.,;")

    def _safe_eval(self, expression: str) -> float:
        """Safely evaluate mathematical expression using AST.

        Args:
            expression: Mathematical expression string

        Returns:
            Evaluated result

        Raises:
            ValueError: If expression contains unsafe operations
        """
        # Pre-process: Convert common math notation to Python
        # Replace ^ with ** for exponentiation (careful not to replace ^^)
        expression = re.sub(r'\^(?!\^)', '**', expression)

        # Parse expression
        try:
            node = ast.parse(expression, mode="eval").body
        except SyntaxError as e:
            raise ValueError(f"Invalid expression: {e}")

        # Evaluate recursively
        return self._eval_node(node)

    def _eval_node(self, node: ast.AST) -> float:
        """Recursively evaluate AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.OPERATORS.get(type(node.op))
            if not op:
                raise ValueError(f"Unsupported operator: {node.op}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.OPERATORS.get(type(node.op))
            if not op:
                raise ValueError(f"Unsupported operator: {node.op}")
            return op(operand)
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name not in self.FUNCTIONS:
                raise ValueError(f"Unsupported function: {func_name}")
            func = self.FUNCTIONS[func_name]
            args = [self._eval_node(arg) for arg in node.args]
            return func(*args)
        else:
            raise ValueError(f"Unsupported expression type: {type(node)}")


class WebSearchServer(BaseMCPServer):
    """MCP server for web search via Tavily API.

    Tavily provides high-quality, AI-optimized search results designed
    for LLM consumption with structured, relevant content.

    Example:
        search = WebSearchServer(api_key="tvly-xxx")
        result = search.execute("latest AI news")

    Cost: ~$0.01 per search (Tavily free tier: 1000 searches/month)
    """

    # Cost per search in USD
    COST_PER_SEARCH = 0.01

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 5,
        search_depth: str = "basic",
        include_answer: bool = True,
        max_content_chars: Optional[int] = None,
        max_total_chars: Optional[int] = None,
        telemetry_collector: Optional[Any] = None,
    ):
        super().__init__(
            name="web_search",
            telemetry_collector=telemetry_collector,
        )
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self.max_results = max_results
        self.search_depth = search_depth
        self.include_answer = include_answer
        self.max_content_chars = max_content_chars
        self.max_total_chars = max_total_chars
        self._client = None

    def _get_client(self):
        """Lazily initialize Tavily client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "TAVILY_API_KEY not set. Get a free API key at https://tavily.com"
                )
            from tavily import TavilyClient
            self._client = TavilyClient(api_key=self.api_key)

        return self._client

    def _execute_impl(self, prompt: str, **params: Any) -> MCPToolResult:
        """Execute web search via Tavily API.

        Args:
            prompt: Search query
            **params: Additional parameters:
                - max_results: Number of results (default: 5)
                - search_depth: 'basic' or 'advanced' (default: 'basic')
                - include_answer: Include AI-generated answer (default: True)

        Returns:
            MCPToolResult with formatted search results
        """
        max_results = params.get("max_results", self.max_results)
        search_depth = params.get("search_depth", self.search_depth)
        include_answer = params.get("include_answer", self.include_answer)
        max_content_chars = params.get("max_content_chars", self.max_content_chars)
        max_total_chars = params.get("max_total_chars", self.max_total_chars)

        # If no API key, return helpful message
        if not self.api_key:
            content = (
                f"[Web search for: {prompt}]\n\n"
                "Web search requires TAVILY_API_KEY environment variable.\n"
                "Get a free API key at: https://tavily.com\n"
                "Then set: export TAVILY_API_KEY='your-key'"
            )
            return MCPToolResult(
                content=content,
                usage={},
                cost_usd=0.0,
                metadata={"tool": "web_search", "error": "no_api_key"},
            )

        try:
            client = self._get_client()
            response = client.search(
                query=prompt,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=include_answer,
            )

            # Format results
            lines = [f"Web search results for: {prompt}\n"]

            # Include AI-generated answer if available
            if include_answer and response.get("answer"):
                lines.append(f"Summary: {response['answer']}\n")

            # Format individual results
            results = response.get("results", [])
            for i, result in enumerate(results, 1):
                title = result.get("title", "No title")
                url = result.get("url", "")
                content_snippet = result.get("content", "")
                if max_content_chars and len(content_snippet) > max_content_chars:
                    content_snippet = content_snippet[:max_content_chars] + "\n... (result truncated)"
                lines.append(f"{i}. {title}")
                lines.append(f"   URL: {url}")
                lines.append(f"   {content_snippet}")
                lines.append("")

            content = "\n".join(lines)
            if max_total_chars and len(content) > max_total_chars:
                content = content[:max_total_chars] + "\n... (search results truncated)"

            return MCPToolResult(
                content=content,
                usage={},
                cost_usd=self.COST_PER_SEARCH,
                metadata={
                    "tool": "web_search",
                    "query": prompt,
                    "num_results": len(results),
                    "search_depth": search_depth,
                },
            )

        except ImportError as e:
            return MCPToolResult(
                content=f"Error: {e}",
                usage={},
                cost_usd=0.0,
                metadata={"tool": "web_search", "error": "import_error"},
            )
        except Exception as e:
            return MCPToolResult(
                content=f"Search error: {type(e).__name__}: {e}",
                usage={},
                cost_usd=0.0,
                metadata={"tool": "web_search", "error": str(e)},
            )


class CodeInterpreterServer(BaseMCPServer):
    """MCP server for Python code execution with optional sandbox isolation.

    Executes Python code in a subprocess with timeout protection.
    Supports bubblewrap (bwrap) for filesystem isolation on Linux.

    Isolation modes:
        - None: Direct subprocess execution (default, for compatibility)
        - "bubblewrap": Linux namespace isolation with read-only root fs
        - "auto": Use bubblewrap if available, fall back to direct execution

    Example:
        interpreter = CodeInterpreterServer(isolation="auto")
        result = interpreter.execute("print([x**2 for x in range(10)])")

    Cost: ~$0.0000083 per second of compute (based on cloud GPU rates)
    """

    # Approximate cost per second of compute (GPU instance rate)
    COST_PER_SECOND = 0.0000083

    # Blocked imports for safety (used in non-isolated mode)
    BLOCKED_IMPORTS = {
        "os.system", "subprocess", "shutil.rmtree", "pathlib.Path.rmdir",
        "eval", "exec", "__import__", "importlib",
    }

    def __init__(
        self,
        timeout: int = 30,
        max_output_length: int = 10000,
        telemetry_collector: Optional[Any] = None,
        isolation: Optional[str] = None,
        allowed_paths: Optional[List[str]] = None,
    ):
        """Initialize code interpreter.

        Args:
            timeout: Maximum execution time in seconds (default: 30)
            max_output_length: Maximum characters to return (default: 10000)
            telemetry_collector: Energy monitor collector
            isolation: Isolation mode - None, "bubblewrap", or "auto"
            allowed_paths: Additional paths to mount read-only in sandbox
        """
        super().__init__(
            name="code_interpreter",
            telemetry_collector=telemetry_collector,
        )
        self.timeout = timeout
        self.max_output_length = max_output_length
        self.allowed_paths = allowed_paths or []

        # Determine isolation mode
        self._use_bubblewrap = False
        if isolation == "bubblewrap":
            if not _check_bubblewrap_available():
                raise RuntimeError(
                    "Bubblewrap (bwrap) not found. Install with: "
                    "apt-get install bubblewrap (Debian/Ubuntu) or "
                    "dnf install bubblewrap (Fedora/RHEL)"
                )
            self._use_bubblewrap = True
        elif isolation == "auto":
            self._use_bubblewrap = _check_bubblewrap_available()

        # Cache Python paths for bubblewrap
        if self._use_bubblewrap:
            self._python_paths = _get_python_lib_paths()

        self.code_extractor = re.compile(r"```[^\n]*\n([\s\S]*?)```", re.DOTALL)

    def _build_bwrap_command(self, script_path: str, sandbox_dir: str) -> List[str]:
        """Build the bubblewrap command for isolated execution."""
        cmd = [
            "bwrap",
            "--unshare-all",
            "--share-net",
            "--die-with-parent",
            "--new-session",
        ]

        # Mount system paths read-only
        for path in self._python_paths:
            if os.path.isdir(path):
                cmd.extend(["--ro-bind", path, path])
            elif os.path.isfile(path):
                cmd.extend(["--ro-bind", path, path])

        # Mount additional allowed paths read-only
        for path in self.allowed_paths:
            if os.path.exists(path):
                real_path = os.path.realpath(path)
                cmd.extend(["--ro-bind", real_path, real_path])

        # Essential virtual filesystems
        cmd.extend([
            "--proc", "/proc",
            "--dev", "/dev",
        ])

        # Create isolated /tmp with our sandbox directory
        cmd.extend([
            "--tmpfs", "/tmp",
            "--bind", sandbox_dir, "/sandbox",
            "--chdir", "/sandbox",
        ])

        # Set environment
        cmd.extend([
            "--setenv", "HOME", "/sandbox",
            "--setenv", "TMPDIR", "/tmp",
            "--setenv", "PATH", "/usr/bin:/bin:/usr/local/bin",
            "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            "--setenv", "PYTHONUNBUFFERED", "1",
        ])

        # Preserve PYTHONPATH for package access
        if os.environ.get("PYTHONPATH"):
            cmd.extend(["--setenv", "PYTHONPATH", os.environ["PYTHONPATH"]])

        # Add the Python command
        cmd.extend([sys.executable, script_path])

        return cmd

    def _preprocess_code(self, code: str) -> str:
        """Preprocess code to extract from markdown blocks."""
        match = self.code_extractor.search(code)
        if match:
            code = match.group(1).strip()
        return code

    def _execute_impl(self, prompt: str, **params: Any) -> MCPToolResult:
        """Execute Python code in subprocess with optional isolation."""
        timeout = params.get("timeout", self.timeout)
        code = prompt.strip()

        # Basic safety check - warn about potentially dangerous operations
        dangerous_patterns = [
            "os.system", "subprocess.run", "subprocess.call", "subprocess.Popen",
            "shutil.rmtree", "os.remove", "os.rmdir", "__import__",
            "open(", "eval(", "exec(",
        ]
        warnings: List[str] = []

        code = self._preprocess_code(code)

        # Only warn in non-isolated mode (these are safe in sandbox)
        if not self._use_bubblewrap:
            for pattern in dangerous_patterns:
                if pattern in code:
                    warnings.append(f"Warning: Code contains '{pattern}'")

        try:
            if self._use_bubblewrap:
                return self._execute_sandboxed(code, timeout, warnings, params)
            else:
                return self._execute_direct(code, timeout, warnings, params)

        except subprocess.TimeoutExpired:
            return MCPToolResult(
                content=f"Error: Code execution timed out after {timeout} seconds",
                usage={},
                cost_usd=timeout * self.COST_PER_SECOND,
                metadata={"tool": "code_interpreter", "error": "timeout"},
            )
        except Exception as e:
            return MCPToolResult(
                content=f"Execution error: {type(e).__name__}: {e}",
                usage={},
                cost_usd=0.0,
                metadata={"tool": "code_interpreter", "error": str(e)},
            )

    def _execute_direct(
        self,
        code: str,
        timeout: int,
        warnings: List[str],
        params: dict,
    ) -> MCPToolResult:
        """Execute code directly in subprocess (no isolation)."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False
        ) as f:
            f.write(code)
            temp_file = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=params.get("working_dir"),
            )
            return self._format_result(result, timeout, warnings, isolated=False)
        finally:
            try:
                os.unlink(temp_file)
            except OSError:
                pass

    def _execute_sandboxed(
        self,
        code: str,
        timeout: int,
        warnings: List[str],
        params: dict,
    ) -> MCPToolResult:
        """Execute code in bubblewrap sandbox with isolated filesystem."""
        with tempfile.TemporaryDirectory(prefix="ipw_sandbox_") as sandbox_dir:
            # Write script to sandbox
            script_path = os.path.join(sandbox_dir, "script.py")
            with open(script_path, "w") as f:
                f.write(code)

            # Build and run sandboxed command
            cmd = self._build_bwrap_command("/sandbox/script.py", sandbox_dir)

            # Optionally disable network
            if not params.get("network", True):
                if "--share-net" in cmd:
                    cmd.remove("--share-net")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            return self._format_result(result, timeout, warnings, isolated=True)

    def _format_result(
        self,
        result: subprocess.CompletedProcess,
        timeout: int,
        warnings: List[str],
        isolated: bool,
    ) -> MCPToolResult:
        """Format subprocess result into MCPToolResult."""
        stdout = result.stdout
        stderr = result.stderr
        return_code = result.returncode

        # Truncate if too long
        if len(stdout) > self.max_output_length:
            stdout = stdout[:self.max_output_length] + "\n... (output truncated)"
        if len(stderr) > self.max_output_length:
            stderr = stderr[:self.max_output_length] + "\n... (output truncated)"

        # Format output
        lines = []
        if warnings:
            lines.extend(warnings)
            lines.append("")

        if stdout:
            lines.append("Output:")
            lines.append(stdout)

        if stderr:
            lines.append("Errors:")
            lines.append(stderr)

        if return_code != 0:
            lines.append(f"\nExit code: {return_code}")

        if not stdout and not stderr:
            lines.append("(No output)")

        content = "\n".join(lines)
        cost_usd = timeout * self.COST_PER_SECOND

        return MCPToolResult(
            content=content,
            usage={},
            cost_usd=cost_usd,
            metadata={
                "tool": "code_interpreter",
                "return_code": return_code,
                "timeout": timeout,
                "warnings": warnings,
                "isolated": isolated,
            },
        )


class ShellServer(BaseMCPServer):
    """MCP server for running shell commands in the active workspace."""

    def __init__(
        self,
        timeout: int = 120,
        max_output_length: int = 20000,
        allowed_dirs: Optional[List[str]] = None,
        telemetry_collector: Optional[Any] = None,
    ):
        super().__init__(name="bash", telemetry_collector=telemetry_collector)
        self.openhands_name = "bash"
        self.timeout = timeout
        self.max_output_length = max_output_length
        self.working_dir = os.getcwd()
        self._ipw_dynamic_allowed_dirs = allowed_dirs is None
        self.allowed_dirs = allowed_dirs or [self.working_dir]

    def _truncate(self, text: str) -> tuple[str, bool]:
        if self.max_output_length and len(text) > self.max_output_length:
            return text[: self.max_output_length] + "\n... (output truncated)", True
        return text, False

    def _execute_impl(self, prompt: str, **params: Any) -> MCPToolResult:
        command = (prompt or params.get("command") or "").strip()
        if not command:
            return MCPToolResult(
                content="Error: No shell command provided",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                cost_usd=0.0,
                metadata={"tool": "bash", "error": "missing_command"},
            )

        timeout = int(params.get("timeout", self.timeout))
        working_dir = params.get("working_dir") or self.working_dir or os.getcwd()
        resolved_cwd = os.path.realpath(str(working_dir))
        if not any(
            resolved_cwd.startswith(os.path.realpath(d)) for d in self.allowed_dirs
        ):
            return MCPToolResult(
                content="Error: Working directory not in allowed directories",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                cost_usd=0.0,
                metadata={
                    "tool": "bash",
                    "error": "permission_denied",
                    "working_dir": resolved_cwd,
                },
            )

        try:
            result = subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",
                cwd=resolved_cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stdout, stdout_truncated = self._truncate(result.stdout or "")
            stderr, stderr_truncated = self._truncate(result.stderr or "")
            parts = [
                f"Exit code: {result.returncode}",
                f"Working directory: {resolved_cwd}",
            ]
            if stdout:
                parts.append(f"STDOUT:\n{stdout}")
            if stderr:
                parts.append(f"STDERR:\n{stderr}")
            if not stdout and not stderr:
                parts.append("(No output)")
            return MCPToolResult(
                content="\n\n".join(parts),
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                cost_usd=0.0,
                metadata={
                    "tool": "bash",
                    "return_code": result.returncode,
                    "working_dir": resolved_cwd,
                    "timeout": timeout,
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                },
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            stdout, stdout_truncated = self._truncate(stdout)
            stderr, stderr_truncated = self._truncate(stderr)
            return MCPToolResult(
                content=(
                    f"Error: Command timed out after {timeout}s\n\n"
                    f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
                ),
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                cost_usd=0.0,
                metadata={
                    "tool": "bash",
                    "error": "timeout",
                    "working_dir": resolved_cwd,
                    "timeout": timeout,
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                },
            )
        except Exception as exc:
            return MCPToolResult(
                content=f"Error: {exc}",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                cost_usd=0.0,
                metadata={"tool": "bash", "error": str(exc), "working_dir": resolved_cwd},
            )


class ThinkServer(BaseMCPServer):
    """MCP server for internal reasoning/scratchpad.

    This is a "thinking" tool that allows the model to break down
    complex problems step-by-step before delegating to other tools.
    It simply returns the input thought process without any processing.

    Example:
        think = ThinkServer()
        result = think.execute("Let me break this down: 1) First... 2) Then...")
    """

    def __init__(self, telemetry_collector: Optional[Any] = None):
        super().__init__(
            name="think",
            telemetry_collector=telemetry_collector,
        )

    def _execute_impl(self, prompt: str, **params: Any) -> MCPToolResult:
        """Process thinking/reasoning (pass-through)."""
        return MCPToolResult(
            content=f"[Thinking]\n{prompt}",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            cost_usd=0.0,
            metadata={"tool": "think"},
        )


class FileReadServer(BaseMCPServer):
    """MCP server for reading file contents.

    Security: Only allows reading files within allowed directories.

    Example:
        reader = FileReadServer(allowed_dirs=["/workspace"])
        result = reader.execute("/workspace/file.txt")
    """

    def __init__(
        self,
        allowed_dirs: Optional[List[str]] = None,
        max_chars: int = 20000,
        telemetry_collector: Optional[Any] = None,
    ):
        super().__init__(name="file_read", telemetry_collector=telemetry_collector)
        self.working_dir = os.getcwd()
        self._ipw_dynamic_allowed_dirs = allowed_dirs is None
        self.allowed_dirs = allowed_dirs or [os.getcwd()]
        self.max_chars = max_chars

    def _extract_docx_text(self, path: str) -> str:
        with zipfile.ZipFile(path) as archive:
            document_xml = archive.read("word/document.xml")
        root = ET.fromstring(document_xml)
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs: list[str] = []
        for paragraph in root.findall(".//w:p", namespace):
            text = "".join(
                node.text or "" for node in paragraph.findall(".//w:t", namespace)
            ).strip()
            if text:
                paragraphs.append(text)
        return "\n".join(paragraphs)

    def _extract_pdf_text(self, path: str) -> str:
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError:
            try:
                from PyPDF2 import PdfReader  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "PDF parsing requires pypdf or PyPDF2. Install one of them, "
                    "or use code_interpreter with a PDF parser available."
                ) from exc

        reader = PdfReader(path)
        pages: list[str] = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        text = "\n\n".join(page.strip() for page in pages if page.strip())
        if not text:
            return (
                "No embedded text found in this PDF. OCR may be required; use "
                "code_interpreter with an OCR package if available."
            )
        return text

    def _truncate_content(self, content: str, max_chars: int) -> tuple[str, bool]:
        if max_chars and len(content) > max_chars:
            return content[:max_chars] + "\n... (file content truncated)", True
        return content, False

    def _execute_impl(self, prompt: str, **params: Any) -> MCPToolResult:
        """Read file contents.

        Args:
            prompt: File path to read
            **params: start_line (1-indexed, default 1), end_line (optional)

        Returns:
            MCPToolResult with file contents or error message
        """
        file_path = prompt.strip()
        file_path = _resolve_workspace_abbrev_path(file_path, self.working_dir)
        if file_path and not os.path.isabs(file_path):
            base_dir = str(params.get("working_dir") or self.working_dir or os.getcwd())
            file_path = os.path.join(base_dir, file_path)
        start_line = params.get("start_line", 1)
        end_line = params.get("end_line")
        max_chars = int(params.get("max_chars", self.max_chars))

        # Security: resolve path and check if within allowed dirs
        try:
            resolved = os.path.realpath(file_path)
            if not any(
                resolved.startswith(os.path.realpath(d)) for d in self.allowed_dirs
            ):
                return MCPToolResult(
                    content="Error: Path not in allowed directories",
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    cost_usd=0.0,
                    metadata={"tool": "file_read", "error": "permission_denied"},
                )

            if resolved.lower().endswith(".docx"):
                content = self._extract_docx_text(resolved)
                lines = content.splitlines(keepends=True)
            elif resolved.lower().endswith(".pdf"):
                content = self._extract_pdf_text(resolved)
                lines = content.splitlines(keepends=True)
            else:
                with open(resolved, "rb") as f:
                    sample = f.read(4096)
                    f.seek(0)
                    if b"\x00" in sample:
                        return MCPToolResult(
                            content=(
                                f"Error: {resolved} appears to be a binary file. "
                                "Use the code_interpreter tool with an appropriate parser "
                                "to extract structured text from it."
                            ),
                            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                            cost_usd=0.0,
                            metadata={"tool": "file_read", "path": resolved, "error": "binary_file"},
                        )
                    raw = f.read()
                lines = raw.decode("utf-8", errors="replace").splitlines(keepends=True)

            # Apply line range (1-indexed)
            start_idx = max(0, start_line - 1)
            if end_line is not None:
                lines = lines[start_idx:end_line]
            else:
                lines = lines[start_idx:]

            content, truncated = self._truncate_content("".join(lines), max_chars)
            return MCPToolResult(
                content=content,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                cost_usd=0.0,
                metadata={
                    "tool": "file_read",
                    "path": resolved,
                    "lines_read": len(lines),
                    "start_line": start_line,
                    "end_line": end_line,
                    "truncated": truncated,
                    "max_chars": max_chars,
                },
            )
        except FileNotFoundError:
            return MCPToolResult(
                content=f"Error: File not found: {file_path}",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                cost_usd=0.0,
                metadata={"tool": "file_read", "error": "file_not_found"},
            )
        except Exception as e:
            return MCPToolResult(
                content=f"Error: {e}",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                cost_usd=0.0,
                metadata={"tool": "file_read", "error": str(e)},
            )


class FileWriteServer(BaseMCPServer):
    """MCP server for writing file contents.

    Security: Only allows writing files within allowed directories.
    Creates parent directories if they don't exist.

    Example:
        writer = FileWriteServer(allowed_dirs=["/workspace"])
        result = writer.execute("/workspace/output.txt", content="Hello, World!")
    """

    def __init__(
        self,
        allowed_dirs: Optional[List[str]] = None,
        telemetry_collector: Optional[Any] = None,
    ):
        super().__init__(name="file_write", telemetry_collector=telemetry_collector)
        self.working_dir = os.getcwd()
        self._ipw_dynamic_allowed_dirs = allowed_dirs is None
        self.allowed_dirs = allowed_dirs or [os.getcwd()]

    def _execute_impl(self, prompt: str, **params: Any) -> MCPToolResult:
        """Write content to file.

        Args:
            prompt: File path to write
            **params: content (required), mode ('w' for overwrite or 'a' for append)

        Returns:
            MCPToolResult with success message or error
        """
        file_path = prompt.strip()
        file_path = _resolve_workspace_abbrev_path(file_path, self.working_dir)
        if file_path and not os.path.isabs(file_path):
            base_dir = str(params.get("working_dir") or self.working_dir or os.getcwd())
            file_path = os.path.join(base_dir, file_path)
        content = params.get("content", "")
        mode = params.get("mode", "w")

        # Validate mode
        if mode not in ("w", "a"):
            return MCPToolResult(
                content=f"Error: Invalid mode '{mode}'. Use 'w' (write) or 'a' (append).",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                cost_usd=0.0,
                metadata={"tool": "file_write", "error": "invalid_mode"},
            )

        # Security: resolve path and check if within allowed dirs
        try:
            # For new files, resolve the parent directory
            parent_dir = os.path.dirname(file_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            resolved = os.path.realpath(file_path)
            if not any(
                resolved.startswith(os.path.realpath(d)) for d in self.allowed_dirs
            ):
                return MCPToolResult(
                    content="Error: Path not in allowed directories",
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    cost_usd=0.0,
                    metadata={"tool": "file_write", "error": "permission_denied"},
                )

            with open(resolved, mode, encoding="utf-8") as f:
                f.write(content)

            action = "appended to" if mode == "a" else "wrote"
            return MCPToolResult(
                content=f"Successfully {action} {len(content)} chars to {resolved}",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                cost_usd=0.0,
                metadata={
                    "tool": "file_write",
                    "path": resolved,
                    "bytes_written": len(content),
                    "mode": mode,
                },
            )
        except Exception as e:
            return MCPToolResult(
                content=f"Error: {e}",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                cost_usd=0.0,
                metadata={"tool": "file_write", "error": str(e)},
            )
