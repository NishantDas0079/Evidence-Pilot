"""Sandboxed tools for the agent under audit.

The agent can only touch files inside one temporary workspace and can only run
the project's tests. It never gets a general shell.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MAX_OUTPUT_CHARS = 4000
TEST_TIMEOUT_S = 30


class ToolError(Exception):
    """A tool failed in a way the agent should see (bad path, missing file...)."""


class Workspace:
    def __init__(self, source_dir: Path):
        self.root = Path(tempfile.mkdtemp(prefix="ep_ws_")).resolve()
        shutil.copytree(source_dir, self.root, dirs_exist_ok=True)

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _resolve(self, relative: str) -> Path:
        if not isinstance(relative, str) or not relative.strip():
            raise ToolError("path must be a non-empty string")
        path = (self.root / relative).resolve()
        if path != self.root and not path.is_relative_to(self.root):
            raise ToolError("path is outside the workspace")
        return path

    def list_files(self) -> str:
        names = sorted(
            str(p.relative_to(self.root)).replace("\\", "/")
            for p in self.root.rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and ".pytest_cache" not in p.parts
        )
        return "\n".join(names) or "(empty)"

    def read_file(self, path: str) -> str:
        target = self._resolve(path)
        if not target.is_file():
            raise ToolError(f"no such file: {path}")
        return target.read_text(encoding="utf-8", errors="replace")[:MAX_OUTPUT_CHARS]

    def write_file(self, path: str, content: str) -> str:
        target = self._resolve(path)
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} characters to {path}"

    def run_tests(self) -> str:
        # Scrubbed environment: the agent's code must never see MODEL_API_KEY.
        env = {
            k: v
            for k, v in os.environ.items()
            if k in {"PATH", "SYSTEMROOT", "TEMP", "TMP", "HOME", "USERPROFILE"}
        }
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
                cwd=self.root,
                env=env,
                capture_output=True,
                text=True,
                timeout=TEST_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return f"exit_code: timeout\ntest run exceeded {TEST_TIMEOUT_S}s and was stopped"
        output = (proc.stdout + proc.stderr)[-MAX_OUTPUT_CHARS:]
        return f"exit_code: {proc.returncode}\n{output}"

    def call(self, name: str, args: dict) -> str:
        """Dispatch one tool call. ToolError becomes a visible tool result."""
        if name == "list_files":
            return self.list_files()
        if name == "read_file":
            return self.read_file(args.get("path"))
        if name == "write_file":
            return self.write_file(args.get("path"), args.get("content"))
        if name == "run_tests":
            return self.run_tests()
        raise ToolError(f"unknown tool: {name}")


def _schema(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


TOOLS = [
    {
        "type": "function",
        "name": "list_files",
        "description": "List all files in the project workspace.",
        "parameters": _schema({}, []),
    },
    {
        "type": "function",
        "name": "read_file",
        "description": "Read a text file from the project workspace.",
        "parameters": _schema({"path": {"type": "string"}}, ["path"]),
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "Create or overwrite a text file in the project workspace.",
        "parameters": _schema(
            {"path": {"type": "string"}, "content": {"type": "string"}},
            ["path", "content"],
        ),
    },
    {
        "type": "function",
        "name": "run_tests",
        "description": "Run the project's test suite with pytest and return the exit code and output.",
        "parameters": _schema({}, []),
    },
]
