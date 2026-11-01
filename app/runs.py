"""Recorded agent runs the public demo can audit. Read-only; nothing here executes code."""
import os
import re
from pathlib import Path

from agent.trace import Trace
from audit.evidence import build_evidence

ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
SUFFIX = ".trace.jsonl"


def trace_dir() -> Path:
    configured = os.getenv("TRACE_DIR")
    return Path(configured) if configured else Path(__file__).resolve().parent.parent / "eval" / "traces"


def _title(run_id: str) -> str:
    return run_id.replace("synthetic_", "").replace("_", " ").replace(".s", " (stopped at step ") + (
        ")" if ".s" in run_id else ""
    )


def _summary(run_id: str, records: list[dict]) -> dict:
    finals = [r for r in records if r["kind"] == "final_report"]
    task = next((r["text"] for r in records if r["kind"] == "task"), "")
    return {
        "id": run_id,
        "title": _title(run_id),
        "synthetic": run_id.startswith("synthetic_"),
        "task": task,
        "report": finals[-1]["text"] if finals else "",
    }


def list_runs() -> list[dict]:
    root = trace_dir()
    if not root.is_dir():
        return []
    out = []
    for path in sorted(root.glob(f"*{SUFFIX}")):
        run_id = path.name[: -len(SUFFIX)]
        if ID_RE.match(run_id):
            out.append(_summary(run_id, Trace.load(path)))
    # real runs first, synthetic after
    return sorted(out, key=lambda r: (r["synthetic"], r["id"]))


def load_run(run_id: str):
    """Return (summary, records) or None. The id is validated against the directory listing."""
    if not ID_RE.match(run_id):
        return None
    root = trace_dir().resolve()
    path = (root / f"{run_id}{SUFFIX}").resolve()
    if path.parent != root or not path.is_file():
        return None
    records = Trace.load(path)
    return _summary(run_id, records), records


def run_items(records: list[dict]) -> list[dict]:
    return [{"id": i.id, "kind": i.kind, "text": i.text} for i in build_evidence(records)]
